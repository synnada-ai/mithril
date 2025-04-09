# Fine-tuning Models with Mithril

This tutorial demonstrates how to fine-tune pre-trained models using Mithril, allowing you to adapt powerful base models to specific domains and tasks with minimal data and computation.

## Overview

In this tutorial, you'll learn:

1. How to load pre-trained models in Mithril
2. How to prepare data for fine-tuning
3. How to implement efficient fine-tuning techniques
4. How to evaluate and save fine-tuned models

## Prerequisites

Before starting, make sure you have the required libraries:

```bash
pip install mithril jax optax numpy torch torchvision matplotlib
```

## Loading Pre-trained Models

Mithril makes it easy to load pre-trained models from popular frameworks like PyTorch and reuse them with different backends. Let's see how to load a pre-trained ResNet model:

```python
import mithril as ml
from mithril.models import resnet18 as ml_resnet18
import torch
import torchvision.models as torch_models
import numpy as np

def load_pretrained_resnet(backend_type="torch"):
    # Create Mithril ResNet model
    mithril_model = ml_resnet18(num_classes=1000)
    
    # Load pre-trained PyTorch model
    torch_model = torch_models.resnet18(pretrained=True)
    torch_state_dict = torch_model.state_dict()
    
    # Create backend based on type
    if backend_type == "torch":
        backend = ml.TorchBackend(dtype=ml.float32)
    elif backend_type == "jax":
        backend = ml.JaxBackend(dtype=ml.float32)
    elif backend_type == "mlx":
        backend = ml.MlxBackend(dtype=ml.float32)
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")
    
    # Compile the Mithril model
    compiled_model = ml.compile(
        model=mithril_model,
        backend=backend,
        shapes={"input": [1, 3, 224, 224]},
        data_keys={"input"},
        jit=True
    )
    
    # Convert PyTorch weights to Mithril format
    mithril_params = {}
    for torch_key, torch_param in torch_state_dict.items():
        # Convert parameter name format
        mithril_key = torch_key.replace(".", "_")
        
        # Check if the parameter exists in the compiled model
        if mithril_key in compiled_model.shapes:
            # Convert to numpy and then to backend array
            mithril_params[mithril_key] = backend.array(torch_param.detach().cpu().numpy())
    
    return compiled_model, mithril_params, backend
```

### Loading a Pre-trained CLIP Model

For multimodal tasks, let's load a pre-trained CLIP model:

```python
import clip as torch_clip
from examples.clip.model import clip as ml_clip

def load_pretrained_clip(backend_type="torch"):
    # Create Mithril CLIP model
    mithril_model = ml_clip(
        embed_dim=512,
        image_resolution=224,
        vision_layers=(3, 4, 6, 3),  # ResNet-50
        vision_width=64,
        vision_patch_size=0,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
    )
    
    # Load pre-trained PyTorch model
    torch_model, _ = torch_clip.load("RN50")
    torch_state_dict = torch_model.state_dict()
    
    # Create backend
    if backend_type == "torch":
        backend = ml.TorchBackend(dtype=ml.float32)
    elif backend_type == "jax":
        backend = ml.JaxBackend(dtype=ml.float32)
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")
    
    # Compile Mithril model
    compiled_model = ml.compile(
        model=mithril_model,
        backend=backend,
        shapes={"image": [1, 3, 224, 224], "text": [1, 77]},
        data_keys={"image", "text"},
        jit=True
    )
    
    # Convert parameters with mapping
    torch_to_mithril_mapping = {
        # Vision encoder
        "visual.conv1.weight": "visual_conv1_weight",
        "visual.bn1.weight": "visual_bn1_weight",
        "visual.bn1.bias": "visual_bn1_bias",
        # ... other mappings
        
        # Text encoder
        "transformer.positional_embedding": "transformer_positional_embedding",
        "transformer.token_embedding.weight": "transformer_token_embedding_weight",
        "transformer.ln_final.weight": "transformer_ln_final_weight",
        "transformer.ln_final.bias": "transformer_ln_final_bias",
        
        # Projection
        "visual.proj": "visual_proj",
        "text_projection": "text_projection",
        "logit_scale": "logit_scale"
    }
    
    # Add layer mappings for ResNet and Transformer blocks
    for i in range(12):  # Transformer layers
        torch_to_mithril_mapping[f"transformer.resblocks.{i}.ln_1.weight"] = f"transformer_resblocks_{i}_ln_1_weight"
        torch_to_mithril_mapping[f"transformer.resblocks.{i}.ln_1.bias"] = f"transformer_resblocks_{i}_ln_1_bias"
        torch_to_mithril_mapping[f"transformer.resblocks.{i}.attn.in_proj_weight"] = f"transformer_resblocks_{i}_attn_in_proj_weight"
        torch_to_mithril_mapping[f"transformer.resblocks.{i}.attn.in_proj_bias"] = f"transformer_resblocks_{i}_attn_in_proj_bias"
        torch_to_mithril_mapping[f"transformer.resblocks.{i}.attn.out_proj.weight"] = f"transformer_resblocks_{i}_attn_out_proj_weight"
        torch_to_mithril_mapping[f"transformer.resblocks.{i}.attn.out_proj.bias"] = f"transformer_resblocks_{i}_attn_out_proj_bias"
        torch_to_mithril_mapping[f"transformer.resblocks.{i}.ln_2.weight"] = f"transformer_resblocks_{i}_ln_2_weight"
        torch_to_mithril_mapping[f"transformer.resblocks.{i}.ln_2.bias"] = f"transformer_resblocks_{i}_ln_2_bias"
        torch_to_mithril_mapping[f"transformer.resblocks.{i}.mlp.c_fc.weight"] = f"transformer_resblocks_{i}_mlp_c_fc_weight"
        torch_to_mithril_mapping[f"transformer.resblocks.{i}.mlp.c_fc.bias"] = f"transformer_resblocks_{i}_mlp_c_fc_bias"
        torch_to_mithril_mapping[f"transformer.resblocks.{i}.mlp.c_proj.weight"] = f"transformer_resblocks_{i}_mlp_c_proj_weight"
        torch_to_mithril_mapping[f"transformer.resblocks.{i}.mlp.c_proj.bias"] = f"transformer_resblocks_{i}_mlp_c_proj_bias"
    
    # Convert parameters
    mithril_params = {}
    for torch_key, mithril_key in torch_to_mithril_mapping.items():
        if torch_key in torch_state_dict:
            # Convert parameter
            param = torch_state_dict[torch_key].detach().cpu().numpy()
            
            # Handle special cases
            if "in_proj" in torch_key:
                # Split the in_proj weight/bias into q, k, v parts
                split_size = param.shape[0] // 3
                param = param.reshape(3, -1, param.shape[-1] if len(param.shape) > 1 else 1)
            
            mithril_params[mithril_key] = backend.array(param)
    
    return compiled_model, mithril_params, backend
```

## Data Preparation for Fine-tuning

### Image Classification Dataset

Let's prepare a custom dataset for fine-tuning an image classifier:

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Find all classes (subdirectories)
        self.classes = sorted([d for d in os.listdir(img_dir) 
                              if os.path.isdir(os.path.join(img_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Find all images
        self.images = []
        self.labels = []
        for class_name in self.classes:
            class_dir = os.path.join(img_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_image_dataloaders(data_dir, batch_size=32):
    # Create datasets
    train_dataset = CustomImageDataset(
        os.path.join(data_dir, 'train')
    )
    val_dataset = CustomImageDataset(
        os.path.join(data_dir, 'val')
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    return train_loader, val_loader, len(train_dataset.classes)
```

### Text Classification Dataset

For fine-tuning language models, let's prepare a text classification dataset:

```python
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import json
import tiktoken

class TextClassificationDataset(Dataset):
    def __init__(self, data_path, tokenizer_name="gpt2", max_length=128):
        self.data_path = data_path
        self.max_length = max_length
        
        # Load tokenizer
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        
        # Load data
        self.texts = []
        self.labels = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for item in data:
            self.texts.append(item['text'])
            self.labels.append(item['label'])
        
        # Get unique labels
        self.unique_labels = sorted(set(self.labels))
        self.label_to_id = {label: i for i, label in enumerate(self.unique_labels)}
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.label_to_id[self.labels[idx]]
        
        # Tokenize text
        tokens = self.tokenizer.encode(text)
        
        # Truncate or pad sequence
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        return torch.tensor(tokens), torch.tensor(label)

def create_text_dataloaders(data_path, batch_size=16):
    # Create datasets
    train_dataset = TextClassificationDataset(
        os.path.join(data_path, 'train.json')
    )
    val_dataset = TextClassificationDataset(
        os.path.join(data_path, 'val.json')
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader, len(train_dataset.unique_labels)
```

## Fine-tuning Techniques

### 1. Full Fine-tuning of Image Classification Models

Let's implement full fine-tuning of a ResNet model on a new dataset:

```python
import optax
import jax
import jax.numpy as jnp
from mithril.models import resnet18, Linear, Model

def finetune_resnet(compiled_model, params, train_loader, val_loader, num_classes, backend, 
                   num_epochs=10, learning_rate=0.001):
    # Create a new classification head for the target dataset
    new_head = Linear(num_classes, use_bias=True)
    
    # Replace the final layer
    model_dict = compiled_model.model.to_dict()
    # Find and replace the final linear layer in the model dictionary
    for idx, layer in enumerate(model_dict['layers']):
        if layer['name'] == 'linear' and idx == len(model_dict['layers']) - 1:
            model_dict['layers'][idx] = new_head.to_dict()
    
    # Create updated model
    new_model = Model.from_dict(model_dict)
    
    # Compile updated model
    new_compiled_model = ml.compile(
        model=new_model,
        backend=backend,
        shapes={"input": [train_loader.batch_size, 3, 224, 224]},
        data_keys={"input"},
        jit=True
    )
    
    # Initialize new layer parameters
    new_params = new_compiled_model.randomize_params()
    
    # Copy parameters from pre-trained model
    for key, value in params.items():
        if key in new_params:
            new_params[key] = value
    
    # Set up optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=learning_rate)
    )
    opt_state = optimizer.init(new_params)
    
    # Create training step function
    def train_step(params, opt_state, images, labels):
        # Forward pass
        outputs = new_compiled_model.evaluate(params, {"input": images})
        logits = outputs["output"]
        
        # Compute cross-entropy loss
        one_hot_labels = jax.nn.one_hot(labels, num_classes)
        loss = -jnp.mean(jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=1))
        
        # Compute gradients
        grads = jax.grad(lambda p: -jnp.mean(jnp.sum(
            one_hot_labels * jax.nn.log_softmax(
                new_compiled_model.evaluate(p, {"input": images})["output"]
            ), axis=1)))(params)
        
        # Update parameters
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        # Calculate accuracy
        predictions = jnp.argmax(logits, axis=1)
        accuracy = jnp.mean(predictions == labels)
        
        return new_params, new_opt_state, loss, accuracy
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        train_loss = 0.0
        train_accuracy = 0.0
        num_batches = 0
        
        for images, labels in train_loader:
            # Convert to backend arrays
            backend_images = backend.array(images.numpy())
            backend_labels = backend.array(labels.numpy())
            
            # Update model
            new_params, opt_state, batch_loss, batch_accuracy = train_step(
                new_params, opt_state, backend_images, backend_labels
            )
            
            train_loss += batch_loss
            train_accuracy += batch_accuracy
            num_batches += 1
        
        # Compute average metrics
        train_loss /= num_batches
        train_accuracy /= num_batches
        
        # Validation
        val_loss = 0.0
        val_accuracy = 0.0
        num_val_batches = 0
        
        for images, labels in val_loader:
            # Convert to backend arrays
            backend_images = backend.array(images.numpy())
            backend_labels = backend.array(labels.numpy())
            
            # Forward pass
            outputs = new_compiled_model.evaluate(new_params, {"input": backend_images})
            logits = outputs["output"]
            
            # Compute loss
            one_hot_labels = jax.nn.one_hot(backend_labels, num_classes)
            batch_loss = -jnp.mean(jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=1))
            
            # Compute accuracy
            predictions = jnp.argmax(logits, axis=1)
            batch_accuracy = jnp.mean(predictions == backend_labels)
            
            val_loss += batch_loss
            val_accuracy += batch_accuracy
            num_val_batches += 1
        
        # Compute average metrics
        val_loss /= num_val_batches
        val_accuracy /= num_val_batches
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.4f}")
    
    return new_compiled_model, new_params
```

### 2. Layer Freezing for Efficient Fine-tuning

For more efficient fine-tuning, we can freeze some layers and only update others:

```python
def freeze_layers_finetune(compiled_model, params, train_loader, val_loader, 
                           num_classes, backend, layers_to_finetune=None,
                           num_epochs=10, learning_rate=0.001):
    """
    Fine-tune a model while freezing specific layers
    
    Args:
        compiled_model: Compiled Mithril model
        params: Pre-trained parameters
        train_loader: Training data loader
        val_loader: Validation data loader
        num_classes: Number of classes in target dataset
        backend: Mithril backend to use
        layers_to_finetune: List of layer name patterns to finetune (others will be frozen)
                           If None, only the final linear layer will be fine-tuned
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
    """
    # Default: finetune only the final layer
    if layers_to_finetune is None:
        layers_to_finetune = ["linear"]
    
    # Create a new classification head
    new_head = Linear(num_classes, use_bias=True)
    
    # Replace the final layer
    model_dict = compiled_model.model.to_dict()
    for idx, layer in enumerate(model_dict['layers']):
        if layer['name'] == 'linear' and idx == len(model_dict['layers']) - 1:
            model_dict['layers'][idx] = new_head.to_dict()
    
    # Create updated model
    new_model = Model.from_dict(model_dict)
    
    # Compile updated model
    new_compiled_model = ml.compile(
        model=new_model,
        backend=backend,
        shapes={"input": [train_loader.batch_size, 3, 224, 224]},
        data_keys={"input"},
        jit=True
    )
    
    # Initialize new parameters
    new_params = new_compiled_model.randomize_params()
    
    # Copy parameters from pre-trained model
    for key, value in params.items():
        if key in new_params:
            new_params[key] = value
    
    # Determine which parameters to update
    trainable_params = {}
    frozen_params = {}
    
    for key, value in new_params.items():
        # Check if this parameter should be fine-tuned
        should_finetune = False
        for pattern in layers_to_finetune:
            if pattern in key:
                should_finetune = True
                break
                
        if should_finetune:
            trainable_params[key] = value
        else:
            frozen_params[key] = value
    
    # Set up optimizer for trainable parameters only
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=learning_rate)
    )
    opt_state = optimizer.init(trainable_params)
    
    # Create training step function that only updates trainable parameters
    def train_step(trainable_params, frozen_params, opt_state, images, labels):
        # Combine parameters
        combined_params = {**trainable_params, **frozen_params}
        
        # Forward pass
        outputs = new_compiled_model.evaluate(combined_params, {"input": images})
        logits = outputs["output"]
        
        # Compute cross-entropy loss
        one_hot_labels = jax.nn.one_hot(labels, num_classes)
        loss = -jnp.mean(jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=1))
        
        # Compute gradients (only for trainable parameters)
        def loss_fn(params):
            # Recombine with frozen params
            combined = {**params, **frozen_params}
            model_output = new_compiled_model.evaluate(combined, {"input": images})["output"]
            return -jnp.mean(jnp.sum(
                one_hot_labels * jax.nn.log_softmax(model_output), axis=1))
            
        grads = jax.grad(loss_fn)(trainable_params)
        
        # Update trainable parameters
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_trainable_params = optax.apply_updates(trainable_params, updates)
        
        # Calculate accuracy
        predictions = jnp.argmax(logits, axis=1)
        accuracy = jnp.mean(predictions == labels)
        
        return new_trainable_params, new_opt_state, loss, accuracy
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        train_loss = 0.0
        train_accuracy = 0.0
        num_batches = 0
        
        for images, labels in train_loader:
            # Convert to backend arrays
            backend_images = backend.array(images.numpy())
            backend_labels = backend.array(labels.numpy())
            
            # Update model
            trainable_params, opt_state, batch_loss, batch_accuracy = train_step(
                trainable_params, frozen_params, opt_state, 
                backend_images, backend_labels
            )
            
            train_loss += batch_loss
            train_accuracy += batch_accuracy
            num_batches += 1
        
        # Calculate average metrics
        train_loss /= num_batches
        train_accuracy /= num_batches
        
        # Validation (using combined parameters)
        val_loss = 0.0
        val_accuracy = 0.0
        num_val_batches = 0
        combined_params = {**trainable_params, **frozen_params}
        
        for images, labels in val_loader:
            # Convert to backend arrays
            backend_images = backend.array(images.numpy())
            backend_labels = backend.array(labels.numpy())
            
            # Forward pass
            outputs = new_compiled_model.evaluate(combined_params, {"input": backend_images})
            logits = outputs["output"]
            
            # Compute loss
            one_hot_labels = jax.nn.one_hot(backend_labels, num_classes)
            batch_loss = -jnp.mean(jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=1))
            
            # Compute accuracy
            predictions = jnp.argmax(logits, axis=1)
            batch_accuracy = jnp.mean(predictions == backend_labels)
            
            val_loss += batch_loss
            val_accuracy += batch_accuracy
            num_val_batches += 1
        
        # Calculate average metrics
        val_loss /= num_val_batches
        val_accuracy /= num_val_batches
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.4f}")
    
    # Return combined parameters
    return new_compiled_model, {**trainable_params, **frozen_params}
```

### 3. Fine-tuning a Transformer Model for Text Classification

Fine-tune a GPT-style model for text classification:

```python
from examples.gpt.model import create_gpt
import jax.numpy as jnp
import optax

def finetune_gpt_for_classification(train_loader, val_loader, num_classes, backend):
    # Create base GPT model
    base_model = create_gpt(
        bias=True,
        block_size=128,
        dims=768,
        num_heads=12,
        num_layers=12,
        vocab_size=50257  # GPT-2 vocabulary size
    )
    
    # Add classification head
    class_model = Model()
    class_model |= base_model(input="input", output="features")
    
    # Take only the last token's representation
    features_shape = class_model.features.shape
    last_token_features = class_model.features[:, -1, :]
    
    # Add classification head
    class_model |= Linear(num_classes)(
        input=last_token_features, output=IOKey("output")
    )
    
    # Compile model
    compiled_model = ml.compile(
        model=class_model,
        backend=backend,
        shapes={"input": [train_loader.batch_size, 128]},
        data_keys={"input"},
        jit=True
    )
    
    # Load pre-trained weights for the base model
    from transformers import GPT2LMHeadModel
    
    torch_model = GPT2LMHeadModel.from_pretrained("gpt2")
    torch_state_dict = torch_model.state_dict()
    
    # Convert weights to Mithril format
    pretrained_params = {}
    for key, value in torch_state_dict.items():
        mithril_key = key.replace(".", "_")
        
        # Handle weight transposition for some layers
        if any(x in key for x in ["attn.c_attn.weight", "attn.c_proj.weight", 
                                 "mlp.c_fc.weight", "mlp.c_proj.weight"]):
            pretrained_params[mithril_key] = backend.array(value.detach().cpu().numpy().T)
        else:
            pretrained_params[mithril_key] = backend.array(value.detach().cpu().numpy())
    
    # Initialize all parameters
    params = compiled_model.randomize_params()
    
    # Copy pre-trained parameters
    for key, value in pretrained_params.items():
        if key in params:
            params[key] = value
    
    # Set up optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=1e-5)  # Use lower learning rate for fine-tuning
    )
    opt_state = optimizer.init(params)
    
    # Training function
    def train_step(params, opt_state, inputs, labels):
        # Forward pass
        outputs = compiled_model.evaluate(params, {"input": inputs})
        logits = outputs["output"]
        
        # Compute cross-entropy loss
        one_hot_labels = jax.nn.one_hot(labels, num_classes)
        loss = -jnp.mean(jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=1))
        
        # Compute gradients
        grads = jax.grad(lambda p: -jnp.mean(jnp.sum(
            one_hot_labels * jax.nn.log_softmax(
                compiled_model.evaluate(p, {"input": inputs})["output"]
            ), axis=1)))(params)
        
        # Update parameters
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        # Calculate accuracy
        predictions = jnp.argmax(logits, axis=1)
        accuracy = jnp.mean(predictions == labels)
        
        return new_params, new_opt_state, loss, accuracy
    
    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        # Training
        train_loss = 0.0
        train_accuracy = 0.0
        num_batches = 0
        
        for inputs, labels in train_loader:
            # Convert to backend arrays
            backend_inputs = backend.array(inputs.numpy())
            backend_labels = backend.array(labels.numpy())
            
            # Update model
            params, opt_state, batch_loss, batch_accuracy = train_step(
                params, opt_state, backend_inputs, backend_labels
            )
            
            train_loss += batch_loss
            train_accuracy += batch_accuracy
            num_batches += 1
        
        # Calculate average metrics
        train_loss /= num_batches
        train_accuracy /= num_batches
        
        # Validation
        val_loss = 0.0
        val_accuracy = 0.0
        num_val_batches = 0
        
        for inputs, labels in val_loader:
            # Convert to backend arrays
            backend_inputs = backend.array(inputs.numpy())
            backend_labels = backend.array(labels.numpy())
            
            # Forward pass
            outputs = compiled_model.evaluate(params, {"input": backend_inputs})
            logits = outputs["output"]
            
            # Compute loss
            one_hot_labels = jax.nn.one_hot(backend_labels, num_classes)
            batch_loss = -jnp.mean(jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=1))
            
            # Compute accuracy
            predictions = jnp.argmax(logits, axis=1)
            batch_accuracy = jnp.mean(predictions == backend_labels)
            
            val_loss += batch_loss
            val_accuracy += batch_accuracy
            num_val_batches += 1
        
        # Calculate average metrics
        val_loss /= num_val_batches
        val_accuracy /= num_val_batches
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.4f}")
    
    return compiled_model, params
```

### 4. Low-Rank Adaptation (LoRA)

LoRA is an efficient fine-tuning method that adds trainable low-rank matrices to pre-trained weights:

```python
from mithril.models import Model, Linear

class LoRALayer(Model):
    def __init__(self, in_dim, out_dim, rank=4, scaling=1.0, name=None):
        super().__init__(name=name)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.scaling = scaling
        
        # Low-rank adaptation matrices
        self.lora_a = Linear(rank, use_bias=False, name="lora_a")
        self.lora_b = Linear(out_dim, use_bias=False, name="lora_b")
        
        # Initialize lora_a with random values
        # Initialize lora_b with zeros for zero init of the LoRA module
    
    def create_model(self):
        model = Model()
        
        # Original input
        input = IOKey("input")
        original_output = IOKey("original_output")
        
        # Apply LoRA matrices
        model |= self.lora_a(input=input, output="lora_a_out")
        model |= self.lora_b(input="lora_a_out", output="lora_b_out")
        
        # Scale output and add to the original output
        lora_output = model.lora_b_out * self.scaling
        model |= Add()(left=original_output, right=lora_output, output=IOKey("output"))
        
        return model

def add_lora_to_model(model, target_modules=None, rank=8, scaling=1.0):
    """
    Add LoRA adapters to a model
    
    Args:
        model: Original Mithril model
        target_modules: List of module name patterns to add LoRA to (e.g., ["attn.c_proj", "mlp.c_proj"])
                       If None, LoRA will be added to all Linear layers
        rank: Rank of the LoRA matrices
        scaling: Scaling factor for LoRA output
    """
    if target_modules is None:
        target_modules = ["linear"]
    
    # Convert model to dictionary
    model_dict = model.to_dict()
    
    # Find Linear layers matching the target patterns
    for idx, layer in enumerate(model_dict['layers']):
        if layer['type'] == 'Linear' and any(pattern in layer['name'] for pattern in target_modules):
            # Get dimensions
            in_dim = layer['in_features']
            out_dim = layer['out_features']
            
            # Create LoRA adapter
            lora_adapter = LoRALayer(in_dim, out_dim, rank, scaling, 
                                     name=f"lora_{layer['name']}")
            
            # Add LoRA adapter after the Linear layer
            # Need to modify the model_dict to insert the LoRA adapter
            
            # This is a simplified approach - in practice, you'd need to 
            # handle the model structure, connections and state more carefully
    
    # Create new model from modified dictionary
    new_model = Model.from_dict(model_dict)
    
    return new_model
```

## Evaluating and Saving Fine-tuned Models

Let's implement functions to evaluate and save our fine-tuned models:

```python
def evaluate_model(compiled_model, params, dataloader, backend, num_classes):
    """Evaluate a model on a dataset"""
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    # For storing predictions for confusion matrix
    all_predictions = []
    all_labels = []
    
    for inputs, labels in dataloader:
        # Convert to backend arrays
        if len(inputs.shape) == 4:  # Image data
            backend_inputs = backend.array(inputs.numpy())
            backend_labels = backend.array(labels.numpy())
            
            # Forward pass
            outputs = compiled_model.evaluate(params, {"input": backend_inputs})
        else:  # Text data
            backend_inputs = backend.array(inputs.numpy())
            backend_labels = backend.array(labels.numpy())
            
            # Forward pass
            outputs = compiled_model.evaluate(params, {"input": backend_inputs})
        
        logits = outputs["output"]
        
        # Compute loss
        one_hot_labels = jax.nn.one_hot(backend_labels, num_classes)
        batch_loss = -jnp.mean(jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=1))
        
        # Compute accuracy
        predictions = jnp.argmax(logits, axis=1)
        batch_accuracy = jnp.mean(predictions == backend_labels)
        
        # Store predictions and labels
        all_predictions.extend(backend.to_numpy(predictions).tolist())
        all_labels.extend(backend.to_numpy(backend_labels).tolist())
        
        total_loss += batch_loss
        total_accuracy += batch_accuracy
        num_batches += 1
    
    # Calculate average metrics
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, avg_accuracy, all_predictions, all_labels

def save_fine_tuned_model(compiled_model, params, backend, save_path):
    """Save a fine-tuned model and its parameters"""
    import os
    import pickle
    import json
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save model architecture
    model_dict = compiled_model.model.to_dict()
    with open(os.path.join(save_path, "model_architecture.json"), "w") as f:
        json.dump(model_dict, f)
    
    # Save parameters (convert to numpy first)
    numpy_params = {k: backend.to_numpy(v) for k, v in params.items()}
    with open(os.path.join(save_path, "model_parameters.pkl"), "wb") as f:
        pickle.dump(numpy_params, f)
    
    print(f"Model saved to {save_path}")

def load_fine_tuned_model(save_path, backend):
    """Load a fine-tuned model and its parameters"""
    import os
    import pickle
    import json
    
    # Load model architecture
    with open(os.path.join(save_path, "model_architecture.json"), "r") as f:
        model_dict = json.load(f)
    
    # Create model from dictionary
    model = Model.from_dict(model_dict)
    
    # Load parameters
    with open(os.path.join(save_path, "model_parameters.pkl"), "rb") as f:
        numpy_params = pickle.load(f)
    
    # Convert parameters to backend format
    params = {k: backend.array(v) for k, v in numpy_params.items()}
    
    return model, params
```

## Visualizing Results

Visualize the performance of fine-tuned models:

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(predictions, true_labels, class_names):
    """Plot confusion matrix for classification results"""
    cm = confusion_matrix(true_labels, predictions, labels=range(len(class_names)))
    
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    """Plot training and validation metrics history"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_model_predictions(compiled_model, params, dataloader, backend, class_names, num_samples=5):
    """Visualize model predictions on sample images"""
    # Get a batch from the dataloader
    inputs, labels = next(iter(dataloader))
    
    # Convert to backend arrays
    backend_inputs = backend.array(inputs.numpy())
    
    # Get predictions
    outputs = compiled_model.evaluate(params, {"input": backend_inputs})
    logits = outputs["output"]
    predictions = backend.to_numpy(backend.argmax(logits, axis=1))
    
    # Convert back to numpy for visualization
    images = inputs.numpy()
    true_labels = labels.numpy()
    
    # Plot images with predictions
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i in range(num_samples):
        ax = axes[i]
        
        # For image data
        if len(images.shape) == 4:
            # Denormalize image
            img = np.transpose(images[i], (1, 2, 0))
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
        else:
            # For text data, show nothing
            ax.axis('off')
            
        # Add title with prediction
        true_class = class_names[true_labels[i]]
        pred_class = class_names[predictions[i]]
        
        title = f"True: {true_class}\nPred: {pred_class}"
        ax.set_title(title, color='green' if true_class == pred_class else 'red')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
```

## Complete Example: Fine-tuning ResNet for Custom Dataset

Let's put everything together in a complete example:

```python
def run_resnet_fine_tuning_example():
    # Load pre-trained ResNet model
    print("Loading pre-trained ResNet model...")
    compiled_model, params, backend = load_pretrained_resnet("torch")
    
    # Load custom dataset
    print("Loading custom dataset...")
    train_loader, val_loader, num_classes = create_image_dataloaders(
        "custom_image_dataset", batch_size=32
    )
    
    # Get class names
    class_names = train_loader.dataset.classes
    
    # Freeze most layers and finetune only selected layers
    print("Fine-tuning model...")
    # Finetune only the final blocks and classification layer
    layers_to_finetune = ["layer4", "linear"]
    
    new_model, new_params = freeze_layers_finetune(
        compiled_model, params, train_loader, val_loader, num_classes, backend,
        layers_to_finetune=layers_to_finetune, num_epochs=10, learning_rate=0.001
    )
    
    # Evaluate the fine-tuned model
    print("Evaluating fine-tuned model...")
    val_loss, val_accuracy, predictions, true_labels = evaluate_model(
        new_model, new_params, val_loader, backend, num_classes
    )
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    
    # Visualize results
    print("Visualizing results...")
    plot_confusion_matrix(predictions, true_labels, class_names)
    
    # Visualize some predictions
    visualize_model_predictions(new_model, new_params, val_loader, backend, 
                               class_names, num_samples=5)
    
    # Save the fine-tuned model
    print("Saving fine-tuned model...")
    save_fine_tuned_model(new_model, new_params, backend, "fine_tuned_resnet")
    
    print("Fine-tuning complete!")
```

## Advanced Fine-tuning Strategies

### Gradual Unfreezing

Gradually unfreeze layers during fine-tuning for better adaptation:

```python
def gradual_unfreeze_finetune(compiled_model, params, train_loader, val_loader, num_classes, backend,
                              layer_groups=None, num_epochs_per_group=3, learning_rate=0.001):
    """
    Fine-tune a model with gradual unfreezing of layer groups
    
    Args:
        compiled_model: Compiled Mithril model
        params: Pre-trained parameters
        train_loader: Training data loader
        val_loader: Validation data loader
        num_classes: Number of classes in target dataset
        backend: Mithril backend to use
        layer_groups: List of layer name pattern groups to gradually unfreeze
                     (first groups are unfrozen first, then progressively more)
        num_epochs_per_group: Number of epochs to train each group configuration
        learning_rate: Base learning rate for optimizer
    """
    # Default layer groups for ResNet (from top to bottom)
    if layer_groups is None:
        layer_groups = [
            ["linear"],  # Output layer only
            ["layer4"],  # Final block
            ["layer3"],  # Second-to-last block
            ["layer2"],  # Third-to-last block
            ["layer1"],  # Early blocks
        ]
    
    # Create a new classification head
    new_head = Linear(num_classes, use_bias=True)
    
    # Replace the final layer
    model_dict = compiled_model.model.to_dict()
    for idx, layer in enumerate(model_dict['layers']):
        if layer['name'] == 'linear' and idx == len(model_dict['layers']) - 1:
            model_dict['layers'][idx] = new_head.to_dict()
    
    # Create updated model
    new_model = Model.from_dict(model_dict)
    
    # Compile updated model
    new_compiled_model = ml.compile(
        model=new_model,
        backend=backend,
        shapes={"input": [train_loader.batch_size, 3, 224, 224]},
        data_keys={"input"},
        jit=True
    )
    
    # Initialize new parameters
    new_params = new_compiled_model.randomize_params()
    
    # Copy parameters from pre-trained model
    for key, value in params.items():
        if key in new_params:
            new_params[key] = value
    
    # Gradually unfreeze and train
    currently_trainable = []
    
    for group_idx, group_patterns in enumerate(layer_groups):
        print(f"Phase {group_idx + 1}/{len(layer_groups)}: Unfreezing {group_patterns}")
        
        # Add new patterns to trainable list
        currently_trainable.extend(group_patterns)
        
        # Determine which parameters to update
        trainable_params = {}
        frozen_params = {}
        
        for key, value in new_params.items():
            # Check if this parameter should be trainable
            should_train = False
            for pattern in currently_trainable:
                if pattern in key:
                    should_train = True
                    break
                    
            if should_train:
                trainable_params[key] = value
            else:
                frozen_params[key] = value
        
        # Adjust learning rate (lower for later phases)
        current_lr = learning_rate / (2 ** group_idx)
        
        # Set up optimizer
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=current_lr)
        )
        opt_state = optimizer.init(trainable_params)
        
        # Train for this phase
        print(f"Training with learning rate: {current_lr}")
        for epoch in range(num_epochs_per_group):
            # Training
            train_loss = 0.0
            train_accuracy = 0.0
            num_batches = 0
            
            for images, labels in train_loader:
                # Convert to backend arrays
                backend_images = backend.array(images.numpy())
                backend_labels = backend.array(labels.numpy())
                
                # Forward pass
                combined_params = {**trainable_params, **frozen_params}
                outputs = new_compiled_model.evaluate(combined_params, {"input": backend_images})
                logits = outputs["output"]
                
                # Compute cross-entropy loss
                one_hot_labels = jax.nn.one_hot(backend_labels, num_classes)
                loss = -jnp.mean(jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=1))
                
                # Compute gradients (only for trainable parameters)
                def loss_fn(params):
                    # Recombine with frozen params
                    combined = {**params, **frozen_params}
                    model_output = new_compiled_model.evaluate(combined, {"input": backend_images})["output"]
                    return -jnp.mean(jnp.sum(
                        one_hot_labels * jax.nn.log_softmax(model_output), axis=1))
                    
                grads = jax.grad(loss_fn)(trainable_params)
                
                # Update trainable parameters
                updates, opt_state = optimizer.update(grads, opt_state)
                trainable_params = optax.apply_updates(trainable_params, updates)
                
                # Calculate accuracy
                predictions = jnp.argmax(logits, axis=1)
                batch_accuracy = jnp.mean(predictions == backend_labels)
                
                train_loss += loss
                train_accuracy += batch_accuracy
                num_batches += 1
            
            # Compute average metrics
            train_loss /= num_batches
            train_accuracy /= num_batches
            
            # Validation
            val_loss, val_accuracy = validate_model(
                new_compiled_model, {**trainable_params, **frozen_params}, 
                val_loader, backend, num_classes
            )
            
            # Print epoch summary
            print(f"Phase {group_idx + 1}, Epoch {epoch+1}/{num_epochs_per_group}, "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Update new_params with the trained parameters
        for key, value in trainable_params.items():
            new_params[key] = value
    
    return new_compiled_model, new_params
```

## Conclusion

In this tutorial, you've learned:

1. How to load pre-trained models in Mithril
2. Different fine-tuning strategies including full fine-tuning and layer freezing
3. How to prepare datasets for custom tasks
4. How to efficiently adapt models to new domains
5. How to evaluate and visualize fine-tuned model performance

Fine-tuning pre-trained models is a powerful technique for transferring knowledge to new tasks with limited data and computational resources. Mithril's flexible architecture makes it easy to implement different fine-tuning strategies across various backends.

## Next Steps

- Experiment with different fine-tuning strategies like knowledge distillation
- Try more advanced techniques like prompt tuning for language models
- Explore domain-specific adaptations for your use cases
- Benchmark performance of different fine-tuning approaches