import yaml
from dataclasses import dataclass
import os
import wandb
from config import Config
from typing import Dict, Any, Literal, Optional
import matplotlib.pyplot as plt

VALID_JOB = ['train', 'test']
VALID_JOB_TYPE = Optional[Literal['train', 'test']]

class WandBLogger:
    def __init__(self, config: Config, job_type: VALID_JOB_TYPE = None, run_name: str = None):
        self.config = config
        self.wandb_config = {
            'models_tested': config.models_tested,
            'evaluation_metrics': config.evaluation_metrics,
            'data': config.data.__dict__,
        }
        self.allow_artifacts = config.wandb.model_versioning
        
        # Add model configs
        for model in config.models_tested:
            model_config = getattr(config, model)
            if model_config:
                self.wandb_config[model] = model_config.__dict__

        self.job_type = job_type

        # Initialize wandb with the new output path structure
        self.run=wandb.init(
            mode=config.wandb.mode,
            entity=config.wandb.entity,
            project=config.wandb.project,
            dir=config.output_paths.run_dir,
            config=self.wandb_config,
            job_type=self.job_type, 
            name=run_name
        )
        self.cm_save_dir = 'confusion_matrix'
        self.roc_save_dir = 'roc_curve'
        self.batch_save_dir = 'batch_examples'
    
    def log_metrics(self, metrics: Dict[str, Any], step: int = None, commit: bool = True):
        """Log metrics to wandb"""
        wandb.log(metrics, step=step, commit=commit)
    
    def log_plot(self, plot_name: str, figure: plt.Figure = None):
        """Log matplotlib figure to wandb"""
        if figure is None:
            figure = plt.gcf()
        wandb.log({plot_name: wandb.Image(figure)})
        plt.close(figure)
    
    def log_model_metrics(self, model_name: str, metrics: Dict[str, Any]):
        """Log model-specific metrics with proper naming"""
        prefixed_metrics = {f"{model_name}/{k}": v for k, v in metrics.items()}
        self.log_metrics(prefixed_metrics)
    
    def log_model_artifact(self, model_name: str, model_path: str):
        """
        Log a model file as a wandb artifact.
        
        Args:
            model_name: Name of the model (will be sanitized for wandb)
            model_path: Path to the saved model file
        """
        
        if not self.allow_artifacts:
            # print(f"{self.allow_artifacts=} Skipping artifact creation.")
            return
        # Sanitize name for wandb (only alphanumeric, dashes, underscores, and dots allowed)
        artifact_name = f"{model_name.lower().replace(' ', '_')}_model"
        
        # Create a new artifact
        artifact = wandb.Artifact(
            name=artifact_name,  # e.g., "naive_bayes_model"
            type="model",        # Type helps organize artifacts
            description=f"Trained {model_name} model"  # Optional description
        )
        
        # Add the model file to the artifact
        artifact.add_file(model_path)
        
        # Log the artifact to wandb
        wandb.log_artifact(artifact)
    
    def log_confusion_matrix(self, model_name: str, y_true, y_pred):
        """Log confusion matrix for a model"""
        wandb.log({
            f"{model_name}/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=['Negative', 'Positive']
            )
        })
    
    def log_roc_curve(self, model_name: str, y_true, y_pred_proba):
        """Log ROC curve for a model"""
        wandb.log({
            f"{model_name}/roc_curve": wandb.plot.roc_curve(
                y_true=y_true,
                y_probas=y_pred_proba,
                labels=['Negative', 'Positive']
            )
        })

    def finish(self):
        """Finish the wandb run"""
        wandb.finish()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

if __name__ == "__main__":
    cfg = Config.from_yaml("config.yml")
    wandb_config = WandBLogger(cfg)
    