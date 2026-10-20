# fl_framework/core/coordinator.py
from __future__ import annotations
from typing import TYPE_CHECKING, List, Any
import time
import os
import logging
from tqdm import tqdm

import torch
from fl_framework.core.hooks import hook_registry, HookType, Context

# For type hinting only
if TYPE_CHECKING:
    from fl_framework.core.server import Server
    from fl_framework.core.client import BaseClient as Client

def client_trigger(clients, context, hook_type: HookType):
    """Hook trigger"""
    for c in clients:
        context.current_client_id = c.client_id
        hook_registry.trigger(hook_type, context)

def client_task(clients, context) -> Any:
    """ This function prepares the task for each client. """
    # Create a per-client context for thread-safety in a real parallel scenario
    for c in clients:
        context.current_client_id = c.client_id

        hook_registry.trigger(HookType.BEFORE_COMPUTE, context)
        
        raw_grad = c.compute_gradients(context)
        # print(raw_grad)
        context.grad[c.client_id] = raw_grad
        
        hook_registry.trigger(HookType.AFTER_COMPUTE, context)
        # The gradient to return may have been modified by the AFTER_COMPUTE hook


class Coordinator:
    """
    The process driver for the federated learning experiment.

    The Coordinator orchestrates the entire training process by following a strict
    sequence of operations: client selection, model distribution, local computation,
    gradient aggregation, and global model update.

    It is the central component that triggers all hooks at the appropriate times.
    It remains completely agnostic to the specific optimization algorithm,
    delegating all complex logic to the Optimizer and other components via the

    hook system. This adherence to the "Optimizer-Centric" design is paramount.
    """

    def __init__(
        self,
        server: Server,
        clients: List[Client],
        num_byzantine: int,
        # resource_manager: BaseResourceManager,
        # Configuration for the experiment run
        num_rounds: int,
        num_round_steps: int,
        save_path: str = None,
    ):
        """
        Initializes the Coordinator.

        Args:
            server (Server): The central server instance.
            clients (List[Client]): A list of all client instances.
            resource_manager (ResourceManager): The manager for parallel execution.
            num_rounds (int): Total number of communication rounds.
            num_round_steps (int): Optimizer iterations per evaluation interval.
        """
        self.server = server
        self.clients = clients
        self.optimizer = server.optimizer # Get optimizer from server
        # self.resource_manager = resource_manager
        self.num_byzantine = num_byzantine  # Number of Byzantine clients

        # Experiment parameters:
        self.save_path = save_path
        self.num_rounds = num_rounds
        self.num_round_steps = num_round_steps
        self.history = {'loss':[], 'metrics':[]}

        # --- Critical Setup Steps ---
        # Register clients with the server
        self.server.register_clients(self.clients)

        # Initialize the master HookContext
        self.context = Context()
        self.context.server = self.server
        self.context.clients = self.clients
        self.context.optimizer = self.server.optimizer
        self.context.aggregator = self.server.aggregator
        self.context.grad = [None for _ in self.clients] # Placeholder for raw gradients
        self.context.num_round_steps = self.num_round_steps
        self.context.num_byzantine = self.num_byzantine

        self.optimizer.register_hooks()

    def run(self, resume_from: str = None) -> None:
        """
        Starts and executes the entire federated learning experiment.
        """
        logging.info("="*30)
        logging.info("Federated Learning Experiment Starting")
        logging.info(f"  - Rounds: {self.num_rounds}")
        logging.info(f"  - Steps per interval: {self.num_round_steps}")
        logging.info("="*30)

        start_time = time.time()

        if resume_from:
            self.resume(resume_from)
            start_round = self.context.current_round+1
        else:
            start_round = 0
            hook_registry.trigger(HookType.BEFORE_RUN, self.context)

        metrics_name = self.server.metrics[0].__name__
        if len(self.history['metrics']) > 0:
            all_metrics = [m[metrics_name] for m in self.history['metrics']]
            self.best_metric, index = torch.max(torch.tensor(all_metrics), dim=0)
            self.best_metric_round = int(index)
            self.best_metric = self.best_metric.item()
            self.best_metric_round -= 1 # exclude a test result before running
        else:
            self.best_metric = -float('inf')
            self.best_metric_round = None

        loss, metrics = self.server.test()
        self.history['loss'].append(loss)
        self.history['metrics'].append(metrics)
        logging.info(f"[Coordinator] Round {-1} Loss: {loss}")
        for k, v in metrics.items():
            logging.info(f"[Coordinator] Round {-1} Metric {k}: {v}")

        for rr in range(start_round, self.num_rounds):
            self.context.current_round = rr
            self._run_round()

            last_pth = [None] + list(filter(lambda x:'latest_state_round_' in x and x.endswith('.pth'), os.listdir(self.save_path)))
            last_pth = last_pth[-1]
            self.save_state(os.path.join(self.save_path, f"latest_state_round_{rr}.pth"))
            if last_pth and last_pth != f"latest_state_round_{rr}.pth":
                os.remove(os.path.join(self.save_path, last_pth))
            
            current_metric = self.history['metrics'][-1][metrics_name]
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.save_state(os.path.join(self.save_path, f"best_state_round_{rr}.pth"))
                logging.info(f"[Coordinator] New best {metrics_name}: {self.best_metric} at round {rr}")
                
                if self.best_metric_round != rr and os.path.exists(os.path.join(self.save_path, f"best_state_round_{self.best_metric_round}.pth")) and self.best_metric_round is not None:
                    os.remove(os.path.join(self.save_path, f"best_state_round_{self.best_metric_round}.pth"))
                self.best_metric_round = rr

            

        hook_registry.trigger(HookType.AFTER_RUN, self.context)
        # self.resource_manager.shutdown()

        end_time = time.time()
        logging.info("\nFederated Learning Experiment Finished.")
        logging.info(f"Total duration: {end_time - start_time:.2f} seconds.")


    def _run_round(self) -> None:
        """Executes a single round."""
        round_num = self.context.current_round
        logging.info(f"\n--- Round {round_num}/{self.num_rounds-1} ---")

        # Trigger Pre-Round Hooks
        hook_registry.trigger(HookType.BEFORE_ROUND_SERVER, self.context)
        client_trigger(self.clients, self.context, HookType.BEFORE_ROUND_CLIENT)
            
        # Training Steps
        for step in tqdm(list(range(self.num_round_steps))):
            self.context.current_step = step
            self._run_step()

        # Trigger Post-Round Hooks
        hook_registry.trigger(HookType.AFTER_ROUND_SERVER, self.context)
        client_trigger(self.clients, self.context, HookType.POST_ROUND_CLIENT)

        # Test
        loss, metrics = self.server.test()
        self.history['loss'].append(loss)
        self.history['metrics'].append(metrics)
        logging.info(f"[Coordinator] Round {round_num} Loss: {loss}")
        for k, v in metrics.items():
            logging.info(f"[Coordinator] Round {round_num} Metric {k}: {v}")


    def _run_step(self) -> None:
        """Compute every worker's gradient, aggregate, and update the model."""
        self.context.grad = [None for _ in self.clients]
        self.context.extra.clear()
        honest = self.clients[:-self.num_byzantine] if self.num_byzantine else self.clients
        byzantine = self.clients[-self.num_byzantine:] if self.num_byzantine else []
        hook_registry.trigger(HookType.BEFORE_STEP_SERVER, self.context)
        client_trigger(self.clients, self.context, HookType.BEFORE_STEP_CLIENT)

        client_task(honest, self.context)
        self.context.all_honest_gradients = [self.context.grad[c.client_id] for c in honest]
        client_task(byzantine, self.context)

        hook_registry.trigger(HookType.BEFORE_AGGREGATE, self.context)
        aggregated_grad = self.server.aggregate_gradients(self.context.grad)
        self.context.aggregated_grad = aggregated_grad
        hook_registry.trigger(HookType.AFTER_AGGREGATE, self.context)
        hook_registry.trigger(HookType.BEFORE_UPDATE, self.context)
        self.optimizer.step(self.server, aggregated_grad)
        hook_registry.trigger(HookType.AFTER_UPDATE, self.context)
        self.server.distribute_model()
        self.context.aggregated_grad = None

        hook_registry.trigger(HookType.AFTER_STEP_SERVER, self.context)
        client_trigger(self.clients, self.context, HookType.AFTER_STEP_CLIENT)
        self.context.all_honest_gradients = None

    def save_state(self, filepath: str) -> None:
        """
        Save the current experiment state to a file for resuming later.
        """
        state = {
            "current_round": self.context.current_round,
            "history": self.history,
            "server_state": self.server.get_state(),
        }
        torch.save(state, filepath)
        logging.info(f"[Coordinator] State saved to {filepath}")

    def resume(self, filepath: str) -> None:
        """
        Resume the experiment from a saved state file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"State file {filepath} not found.")
        state = torch.load(filepath, map_location=self.server.device, weights_only=False)
        self.context.current_round = state["current_round"]
        self.server.set_state(state["server_state"])
        self.history = state['history']
        logging.info(f"[Coordinator] State loaded from {filepath}")
