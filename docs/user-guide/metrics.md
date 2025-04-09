# Metrics

This guide covers how to implement and use metrics for evaluating model performance in Mithril.

## Overview

Metrics are quantitative measures used to evaluate model performance. In Mithril:

- Metrics are computed outside the model
- Metrics can be used during training and evaluation
- Common metrics are provided through utility functions
- Custom metrics can be easily implemented

## Common Metrics

Mithril provides implementations of common metrics in the `mithril.utils.metrics` module.

### Classification Metrics

#### Accuracy

```python
from mithril.utils.metrics import accuracy

# For multi-class classification with argmax
def compute_accuracy(predictions, targets):
    """Compute accuracy for class predictions."""
    # If predictions are probabilities (e.g., softmax outputs)
    pred_classes = backend.argmax(predictions, axis=1)
    target_classes = backend.argmax(targets, axis=1) if targets.ndim > 1 else targets
    
    # Compute accuracy
    acc = accuracy(pred_classes, target_classes)
    return acc

# Usage in training loop
outputs = compiled_model.evaluate(params, batch_inputs)
predictions = outputs["output"]
batch_accuracy = compute_accuracy(predictions, batch_targets)
print(f"Batch accuracy: {batch_accuracy:.4f}")
```

#### Precision, Recall, F1 Score

```python
from mithril.utils.metrics import precision, recall, f1_score

# For binary classification
def compute_binary_metrics(predictions, targets, threshold=0.5):
    """Compute precision, recall, and F1 score for binary classification."""
    # Convert probabilities to binary predictions
    binary_preds = backend.cast(predictions > threshold, backend.float32)
    
    # Compute metrics
    prec = precision(binary_preds, targets)
    rec = recall(binary_preds, targets)
    f1 = f1_score(binary_preds, targets)
    
    return {
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }

# For multi-class classification
def compute_multiclass_metrics(predictions, targets):
    """Compute precision, recall, and F1 score for multi-class classification."""
    # Convert to class predictions
    pred_classes = backend.argmax(predictions, axis=1)
    target_classes = backend.argmax(targets, axis=1) if targets.ndim > 1 else targets
    
    # Get number of classes
    num_classes = predictions.shape[1]
    
    # Compute metrics for each class
    class_metrics = []
    for class_idx in range(num_classes):
        # Create binary indicators for this class
        class_preds = backend.cast(pred_classes == class_idx, backend.float32)
        class_targets = backend.cast(target_classes == class_idx, backend.float32)
        
        # Compute metrics
        prec = precision(class_preds, class_targets)
        rec = recall(class_preds, class_targets)
        f1 = f1_score(class_preds, class_targets)
        
        class_metrics.append({
            "class": class_idx,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })
    
    # Compute macro-averaged metrics
    macro_precision = sum(m["precision"] for m in class_metrics) / num_classes
    macro_recall = sum(m["recall"] for m in class_metrics) / num_classes
    macro_f1 = sum(m["f1_score"] for m in class_metrics) / num_classes
    
    return {
        "class_metrics": class_metrics,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1
    }
```

#### Confusion Matrix

```python
from mithril.utils.metrics import confusion_matrix

def compute_confusion_matrix(predictions, targets, num_classes):
    """Compute the confusion matrix for classification."""
    # Convert to class predictions
    pred_classes = backend.argmax(predictions, axis=1)
    target_classes = backend.argmax(targets, axis=1) if targets.ndim > 1 else targets
    
    # Compute confusion matrix
    cm = confusion_matrix(pred_classes, target_classes, num_classes)
    return cm

# Visualize confusion matrix
def plot_confusion_matrix(cm, class_names=None):
    """Plot a confusion matrix."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert to NumPy if needed
    if hasattr(cm, "numpy"):
        cm = cm.numpy()
    elif hasattr(cm, "detach"):
        cm = cm.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Show all ticks and label them
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xlabel='Predicted label',
           ylabel='True label',
           title='Confusion Matrix')
    
    # Label axes with class names if provided
    if class_names is not None:
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    return fig
```

#### ROC AUC and PR AUC

```python
from mithril.utils.metrics import roc_auc, pr_auc

def compute_roc_pr_metrics(predictions, targets):
    """Compute ROC AUC and PR AUC for binary classification."""
    # For binary classification with sigmoid outputs
    roc_auc_score = roc_auc(predictions, targets)
    pr_auc_score = pr_auc(predictions, targets)
    
    return {
        "roc_auc": roc_auc_score,
        "pr_auc": pr_auc_score
    }

# For multi-class ROC AUC (one-vs-rest approach)
def compute_multiclass_roc_auc(predictions, targets):
    """Compute ROC AUC for each class in multi-class setting."""
    # Convert one-hot targets if needed
    if targets.ndim > 1:
        target_classes = backend.argmax(targets, axis=1)
    else:
        target_classes = targets
    
    # Number of classes
    num_classes = predictions.shape[1]
    
    # Compute ROC AUC for each class
    class_roc_aucs = []
    for class_idx in range(num_classes):
        # Create binary indicators for this class
        class_probs = predictions[:, class_idx]
        class_targets = backend.cast(target_classes == class_idx, backend.float32)
        
        # Compute ROC AUC
        class_roc_auc = roc_auc(class_probs, class_targets)
        class_roc_aucs.append(class_roc_auc)
    
    # Compute macro-averaged ROC AUC
    macro_roc_auc = sum(class_roc_aucs) / num_classes
    
    return {
        "class_roc_aucs": class_roc_aucs,
        "macro_roc_auc": macro_roc_auc
    }
```

### Regression Metrics

#### Mean Squared Error (MSE)

```python
from mithril.utils.metrics import mse

def compute_mse(predictions, targets):
    """Compute Mean Squared Error."""
    return mse(predictions, targets)
```

#### Mean Absolute Error (MAE)

```python
from mithril.utils.metrics import mae

def compute_mae(predictions, targets):
    """Compute Mean Absolute Error."""
    return mae(predictions, targets)
```

#### Root Mean Squared Error (RMSE)

```python
from mithril.utils.metrics import rmse

def compute_rmse(predictions, targets):
    """Compute Root Mean Squared Error."""
    return rmse(predictions, targets)
```

#### R-squared (Coefficient of Determination)

```python
from mithril.utils.metrics import r_squared

def compute_r_squared(predictions, targets):
    """Compute R-squared score."""
    return r_squared(predictions, targets)
```

### Ranking Metrics

#### Mean Average Precision (MAP)

```python
from mithril.utils.metrics import mean_average_precision

def compute_map(predictions, targets, k=None):
    """Compute Mean Average Precision."""
    return mean_average_precision(predictions, targets, k)
```

#### Normalized Discounted Cumulative Gain (NDCG)

```python
from mithril.utils.metrics import ndcg

def compute_ndcg(predictions, targets, k=None):
    """Compute Normalized Discounted Cumulative Gain."""
    return ndcg(predictions, targets, k)
```

## Custom Metrics

You can implement custom metrics tailored to your specific requirements:

```python
def balanced_accuracy(predictions, targets):
    """Compute balanced accuracy for imbalanced classification."""
    # Convert to class predictions
    pred_classes = backend.argmax(predictions, axis=1)
    target_classes = backend.argmax(targets, axis=1) if targets.ndim > 1 else targets
    
    # Get unique classes
    unique_classes = backend.unique(target_classes)
    num_classes = len(unique_classes)
    
    # Compute per-class accuracy
    class_accuracies = []
    for class_idx in unique_classes:
        # Get indices for this class
        class_mask = target_classes == class_idx
        
        # Skip if no examples of this class
        if backend.sum(class_mask) == 0:
            continue
        
        # Compute accuracy for this class
        class_correct = backend.sum(
            backend.logical_and(class_mask, pred_classes == class_idx)
        )
        class_total = backend.sum(class_mask)
        class_acc = class_correct / class_total
        class_accuracies.append(class_acc)
    
    # Average class accuracies
    bal_acc = sum(class_accuracies) / len(class_accuracies)
    return bal_acc
```

## Using Metrics in Training Loops

### Basic Metrics Tracking

```python
import mithril as ml
from mithril.backends import JaxBackend
from mithril.utils.metrics import accuracy, precision, recall, f1_score

# Create and compile a model
model = ml.Model()
# ... define your model ...
backend = JaxBackend()
compiled_model = ml.compile(model, backend)

# Initialize parameters
params = compiled_model.get_parameters()

# Training metrics
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Training metrics for this epoch
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    epoch_precision = 0.0
    epoch_recall = 0.0
    epoch_f1 = 0.0
    num_batches = 0
    
    # Training loop
    for batch_inputs, batch_targets in train_data_loader(batch_size):
        # Forward pass
        outputs = compiled_model.evaluate(params, batch_inputs)
        predictions = outputs["output"]
        
        # Compute loss
        loss_value = cross_entropy_loss(predictions, batch_targets)
        epoch_loss += loss_value
        
        # Compute metrics
        pred_classes = backend.argmax(predictions, axis=1)
        target_classes = backend.argmax(batch_targets, axis=1)
        
        batch_accuracy = accuracy(pred_classes, target_classes)
        batch_precision = precision(pred_classes, target_classes)
        batch_recall = recall(pred_classes, target_classes)
        batch_f1 = f1_score(pred_classes, target_classes)
        
        epoch_accuracy += batch_accuracy
        epoch_precision += batch_precision
        epoch_recall += batch_recall
        epoch_f1 += batch_f1
        
        num_batches += 1
        
        # Update parameters (training step)
        # ...
    
    # Compute average metrics for the epoch
    avg_loss = epoch_loss / num_batches
    avg_accuracy = epoch_accuracy / num_batches
    avg_precision = epoch_precision / num_batches
    avg_recall = epoch_recall / num_batches
    avg_f1 = epoch_f1 / num_batches
    
    # Store training metrics
    train_losses.append(avg_loss)
    train_accuracies.append(avg_accuracy)
    
    # Validation
    val_loss, val_accuracy = evaluate(compiled_model, params, val_data_loader)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    # Print epoch summary
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, "
          f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
```

### Creating a Metrics Tracker

For more organized metrics tracking, create a metrics tracking class:

```python
class MetricsTracker:
    """Tracks and computes metrics during training and evaluation."""
    
    def __init__(self, backend, metric_fns=None):
        """Initialize the metrics tracker.
        
        Args:
            backend: The backend instance
            metric_fns: Dictionary of metric name to metric function
        """
        self.backend = backend
        self.metric_fns = metric_fns or {}
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.batch_count = 0
        self.metrics_sum = {name: 0.0 for name in self.metric_fns}
        self.metrics_values = {name: [] for name in self.metric_fns}
    
    def update(self, predictions, targets):
        """Update metrics with a new batch of predictions and targets."""
        self.batch_count += 1
        
        # Compute and update each metric
        for name, metric_fn in self.metric_fns.items():
            value = metric_fn(predictions, targets)
            self.metrics_sum[name] += value
            self.metrics_values[name].append(value)
    
    def result(self):
        """Get the average metrics."""
        if self.batch_count == 0:
            return {name: 0.0 for name in self.metric_fns}
        
        return {name: self.metrics_sum[name] / self.batch_count
                for name in self.metric_fns}
    
    def results_per_batch(self):
        """Get metrics for each batch."""
        return self.metrics_values

# Usage example
from mithril.utils.metrics import accuracy, precision, recall, f1_score

# Create metric functions
def compute_accuracy(predictions, targets):
    pred_classes = backend.argmax(predictions, axis=1)
    target_classes = backend.argmax(targets, axis=1)
    return accuracy(pred_classes, target_classes)

def compute_precision(predictions, targets):
    pred_classes = backend.argmax(predictions, axis=1)
    target_classes = backend.argmax(targets, axis=1)
    return precision(pred_classes, target_classes)

# Create a metrics tracker
metrics_tracker = MetricsTracker(
    backend,
    metric_fns={
        "accuracy": compute_accuracy,
        "precision": compute_precision,
    }
)

# Use in training loop
for epoch in range(num_epochs):
    # Reset metrics for each epoch
    metrics_tracker.reset()
    
    for batch_inputs, batch_targets in train_data_loader(batch_size):
        # Forward pass
        outputs = compiled_model.evaluate(params, batch_inputs)
        predictions = outputs["output"]
        
        # Update metrics
        metrics_tracker.update(predictions, batch_targets)
        
        # Update parameters (training step)
        # ...
    
    # Get average metrics for the epoch
    avg_metrics = metrics_tracker.result()
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Accuracy: {avg_metrics['accuracy']:.4f}, "
          f"Precision: {avg_metrics['precision']:.4f}")
```

## Evaluating Models with Metrics

### Comprehensive Evaluation Function

```python
def evaluate_model(model, params, data_loader, metrics=None):
    """Evaluate a model on the given data with multiple metrics.
    
    Args:
        model: Compiled Mithril model
        params: Model parameters
        data_loader: Function that yields (inputs, targets) batches
        metrics: Dictionary of metric name to metric function
    
    Returns:
        Dictionary of metric name to average metric value
    """
    metrics = metrics or {}
    batch_count = 0
    metrics_sum = {name: 0.0 for name in metrics}
    all_predictions = []
    all_targets = []
    
    # Disable gradient tracking during evaluation
    with backend.no_grad():
        for batch_inputs, batch_targets in data_loader:
            # Forward pass
            outputs = model.evaluate(params, batch_inputs)
            predictions = outputs["output"]
            
            # Store predictions and targets for computing global metrics
            all_predictions.append(predictions)
            all_targets.append(batch_targets)
            
            # Update per-batch metrics
            for name, metric_fn in metrics.items():
                if name.startswith('batch_'):  # Only compute batch metrics
                    value = metric_fn(predictions, batch_targets)
                    metrics_sum[name] += value
            
            batch_count += 1
    
    # Concatenate all predictions and targets
    if all_predictions and all_targets:
        all_predictions = backend.concatenate(all_predictions, axis=0)
        all_targets = backend.concatenate(all_targets, axis=0)
        
        # Compute global metrics
        for name, metric_fn in metrics.items():
            if not name.startswith('batch_'):  # Only compute global metrics
                metrics_sum[name] = metric_fn(all_predictions, all_targets)
    
    # Compute average for batch metrics
    results = {}
    for name, value_sum in metrics_sum.items():
        if name.startswith('batch_'):
            results[name] = value_sum / batch_count
        else:
            results[name] = value_sum
    
    return results

# Example usage
from mithril.utils.metrics import accuracy, precision, recall, f1_score, confusion_matrix

# Define metric functions
def compute_accuracy(predictions, targets):
    pred_classes = backend.argmax(predictions, axis=1)
    target_classes = backend.argmax(targets, axis=1)
    return accuracy(pred_classes, target_classes)

def compute_precision(predictions, targets):
    pred_classes = backend.argmax(predictions, axis=1)
    target_classes = backend.argmax(targets, axis=1)
    return precision(pred_classes, target_classes)

def compute_cm(predictions, targets):
    pred_classes = backend.argmax(predictions, axis=1)
    target_classes = backend.argmax(targets, axis=1)
    return confusion_matrix(pred_classes, target_classes, num_classes=10)

# Define metrics for evaluation
eval_metrics = {
    "accuracy": compute_accuracy,
    "precision": compute_precision,
    "recall": lambda p, t: recall(backend.argmax(p, axis=1), backend.argmax(t, axis=1)),
    "f1_score": lambda p, t: f1_score(backend.argmax(p, axis=1), backend.argmax(t, axis=1)),
    "confusion_matrix": compute_cm,
    "batch_loss": lambda p, t: cross_entropy_loss(p, t),  # Per-batch metric
}

# Evaluate the model
evaluation_results = evaluate_model(compiled_model, params, val_data_loader, eval_metrics)

# Print results
print("Evaluation Results:")
for name, value in evaluation_results.items():
    if name != "confusion_matrix":  # Skip printing large matrices
        print(f"{name}: {value:.4f}")

# Plot confusion matrix
cm = evaluation_results["confusion_matrix"]
plot_confusion_matrix(cm, class_names=class_names)
```

## Visualizing Metrics

### Plotting Metrics Over Time

```python
import matplotlib.pyplot as plt

def plot_metrics_history(metrics_history, figsize=(12, 8)):
    """Plot training and validation metrics over time.
    
    Args:
        metrics_history: Dict with metric names as keys and lists of values as values.
                        Should include 'epoch' list and pairs of train/val metrics.
        figsize: Figure size as (width, height) tuple.
    """
    # Get list of unique metric names (excluding 'epoch' and removing train/val prefixes)
    metric_names = set()
    for name in metrics_history.keys():
        if name != 'epoch':
            # Remove train_ or val_ prefix
            if name.startswith('train_'):
                base_name = name[6:]
                metric_names.add(base_name)
            elif name.startswith('val_'):
                base_name = name[4:]
                metric_names.add(base_name)
    
    # Sort metric names for consistent ordering
    metric_names = sorted(metric_names)
    
    # Calculate grid dimensions
    n_metrics = len(metric_names)
    n_cols = min(3, n_metrics)  # At most 3 columns
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows * n_cols > 1:  # If multiple subplots
        axes = axes.flatten()
    else:  # If only one subplot
        axes = [axes]
    
    # Plot each metric
    for i, metric_name in enumerate(metric_names):
        ax = axes[i]
        
        # Get train and validation metric names
        train_name = f'train_{metric_name}'
        val_name = f'val_{metric_name}'
        
        # Plot train metric if available
        if train_name in metrics_history:
            ax.plot(metrics_history['epoch'], metrics_history[train_name], 
                   'b-', marker='o', label=f'Train {metric_name}')
        
        # Plot validation metric if available
        if val_name in metrics_history:
            ax.plot(metrics_history['epoch'], metrics_history[val_name], 
                   'r-', marker='x', label=f'Val {metric_name}')
        
        # Set labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} over time')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    return fig

# Example usage
metrics_history = {
    'epoch': list(range(1, 11)),  # Epochs 1-10
    'train_loss': [2.3, 1.9, 1.5, 1.3, 1.1, 0.9, 0.8, 0.7, 0.65, 0.6],
    'val_loss': [2.4, 2.0, 1.7, 1.5, 1.3, 1.2, 1.1, 1.05, 1.0, 1.0],
    'train_accuracy': [0.2, 0.4, 0.55, 0.65, 0.7, 0.75, 0.8, 0.82, 0.85, 0.86],
    'val_accuracy': [0.15, 0.35, 0.5, 0.6, 0.65, 0.68, 0.7, 0.71, 0.71, 0.72],
    'train_f1_score': [0.18, 0.38, 0.52, 0.63, 0.68, 0.73, 0.78, 0.8, 0.83, 0.84],
    'val_f1_score': [0.13, 0.33, 0.47, 0.57, 0.62, 0.65, 0.67, 0.68, 0.68, 0.69],
}

plot_metrics_history(metrics_history)
plt.savefig('metrics_history.png')
plt.show()
```

### ROC and Precision-Recall Curves

```python
def plot_roc_curve(predictions, targets, num_classes=None):
    """Plot ROC curve for binary or multi-class classification.
    
    Args:
        predictions: Predicted probabilities
        targets: Target values
        num_classes: Number of classes for multi-class classification
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import roc_curve, auc
    
    # Convert to NumPy arrays if needed
    if hasattr(predictions, "numpy"):
        predictions = predictions.numpy()
    elif hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu().numpy()
    
    if hasattr(targets, "numpy"):
        targets = targets.numpy()
    elif hasattr(targets, "detach"):
        targets = targets.detach().cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    
    # Binary classification
    if num_classes is None or num_classes <= 2:
        # Ensure predictions are probabilities between 0 and 1
        if predictions.ndim > 1:
            predictions = predictions[:, 1]  # Use probability of positive class
        
        # Ensure targets are binary
        if targets.ndim > 1:
            targets = targets[:, 1]
        
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(targets, predictions)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
    
    # Multi-class classification
    else:
        # Ensure targets are one-hot encoded or class indices
        if targets.ndim == 1:
            # Convert to one-hot
            target_indices = targets
            targets = np.zeros((len(targets), num_classes))
            targets[np.arange(len(target_indices)), target_indices] = 1
        
        # Compute ROC curve and AUC for each class
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(targets[:, i], predictions[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
    
    return plt.gcf()

def plot_precision_recall_curve(predictions, targets, num_classes=None):
    """Plot precision-recall curve for binary or multi-class classification.
    
    Args:
        predictions: Predicted probabilities
        targets: Target values
        num_classes: Number of classes for multi-class classification
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    # Convert to NumPy arrays if needed
    if hasattr(predictions, "numpy"):
        predictions = predictions.numpy()
    elif hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu().numpy()
    
    if hasattr(targets, "numpy"):
        targets = targets.numpy()
    elif hasattr(targets, "detach"):
        targets = targets.detach().cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    
    # Binary classification
    if num_classes is None or num_classes <= 2:
        # Ensure predictions are probabilities between 0 and 1
        if predictions.ndim > 1:
            predictions = predictions[:, 1]  # Use probability of positive class
        
        # Ensure targets are binary
        if targets.ndim > 1:
            targets = targets[:, 1]
        
        # Compute precision-recall curve and average precision
        precision, recall, _ = precision_recall_curve(targets, predictions)
        ap = average_precision_score(targets, predictions)
        
        # Plot precision-recall curve
        plt.plot(recall, precision, lw=2, label=f'PR curve (AP = {ap:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
    
    # Multi-class classification
    else:
        # Ensure targets are one-hot encoded or class indices
        if targets.ndim == 1:
            # Convert to one-hot
            target_indices = targets
            targets = np.zeros((len(targets), num_classes))
            targets[np.arange(len(target_indices)), target_indices] = 1
        
        # Compute precision-recall curve and average precision for each class
        for i in range(num_classes):
            precision, recall, _ = precision_recall_curve(targets[:, i], predictions[:, i])
            ap = average_precision_score(targets[:, i], predictions[:, i])
            plt.plot(recall, precision, lw=2, label=f'Class {i} (AP = {ap:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
    
    return plt.gcf()
```

## Metric Utilities

### Moving Average Metrics

For smoother metrics during training:

```python
class MovingAverageMetric:
    """Tracks a metric using a moving average."""
    
    def __init__(self, window_size=10):
        """Initialize with a window size for the moving average."""
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        """Reset the metric."""
        self.values = []
    
    def update(self, value):
        """Update the metric with a new value."""
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
    
    def result(self):
        """Get the current moving average."""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

# Usage
train_loss_metric = MovingAverageMetric(window_size=20)
train_acc_metric = MovingAverageMetric(window_size=20)

# In training loop
for batch_inputs, batch_targets in train_data_loader(batch_size):
    # Forward pass, compute loss and accuracy
    # ...
    
    # Update metrics
    train_loss_metric.update(loss_value)
    train_acc_metric.update(accuracy_value)
    
    # Print current metrics
    print(f"Moving Avg Loss: {train_loss_metric.result():.4f}, "
          f"Moving Avg Accuracy: {train_acc_metric.result():.4f}")
```

### Early Stopping with Metrics

Use metrics for early stopping:

```python
class EarlyStopping:
    """Early stopping based on validation metrics."""
    
    def __init__(self, metric='val_loss', patience=10, min_delta=0, mode='min'):
        """Initialize early stopping.
        
        Args:
            metric: Metric name to monitor
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for metrics to minimize (like loss), 'max' for metrics to maximize
        """
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.reset()
    
    def reset(self):
        """Reset early stopping state."""
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.wait = 0
        self.best_epoch = 0
        self.stopped_epoch = 0
    
    def should_stop(self, epoch, metrics):
        """Check if training should stop based on validation metric.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metric values
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.metric not in metrics:
            return False
        
        value = metrics[self.metric]
        
        if self.mode == 'min':
            # Check if value decreased by at least min_delta
            if value < self.best_value - self.min_delta:
                self.best_value = value
                self.best_epoch = epoch
                self.wait = 0
            else:
                self.wait += 1
        else:  # mode == 'max'
            # Check if value increased by at least min_delta
            if value > self.best_value + self.min_delta:
                self.best_value = value
                self.best_epoch = epoch
                self.wait = 0
            else:
                self.wait += 1
        
        # Check if patience is exhausted
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            return True
        
        return False

# Usage example
early_stopping = EarlyStopping(
    metric='val_loss',
    patience=5,
    min_delta=0.001,
    mode='min'
)

# In training loop
for epoch in range(num_epochs):
    # Train for one epoch
    train_metrics = train_epoch(model, params, train_data)
    
    # Evaluate on validation data
    val_metrics = evaluate_model(model, params, val_data)
    
    # Combine metrics
    metrics = {**train_metrics, **val_metrics}
    
    # Check early stopping
    if early_stopping.should_stop(epoch, metrics):
        print(f"Early stopping at epoch {epoch}")
        print(f"Best value was {early_stopping.best_value:.4f} at epoch {early_stopping.best_epoch}")
        break
```

## Best Practices

1. **Choose appropriate metrics** for your task:
   - Classification: accuracy, precision, recall, F1, ROC AUC
   - Regression: MSE, MAE, RMSE, R-squared
   - Ranking: MAP, NDCG

2. **Track multiple metrics** to get a comprehensive view:
   - Different metrics emphasize different aspects of performance
   - Some metrics are more sensitive to class imbalance

3. **Monitor both training and validation metrics**:
   - Large gap between training and validation metrics indicates overfitting
   - Use validation metrics for model selection and early stopping

4. **Consider business requirements**:
   - Precision may be more important than recall in some applications
   - Cost of false positives vs. false negatives

5. **Visualize metrics** to better understand model behavior:
   - Plot metrics over time to track training progress
   - Examine confusion matrices to identify problematic classes
   - ROC and precision-recall curves provide insights beyond single numbers

6. **Use moving averages** for smoother progress tracking:
   - Especially helpful for noisy metrics
   - Helps identify trends without getting distracted by fluctuations

7. **Implement early stopping** based on validation metrics:
   - Prevents overfitting
   - Saves computation time
   - Base stopping decisions on the metric most relevant to your goal