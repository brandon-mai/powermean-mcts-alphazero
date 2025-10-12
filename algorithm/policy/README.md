# Policy & Model Implementation Guide
This guide describes how to implement and extend the algorithm part in this repository. It is divided into two main sections: `Policy` and `Model`. The Model section is currently left blank for future instructions.
## Policy
All new `Policy` must inherite from `AbtractPolicy` class and follow these requirements to ensure compatibility and consistency.

### Required Attributes 
1. `self.model`: The neural network model object used for policy and value prediction. This should be compatible with PyTorch.
3. `self.game`: The game environment object.
4. `self.args`: Others attributes, defined specific for class.

### Required Methods
1. `learn(self)`: Main training loop.
2. `play(self)`:  
---

