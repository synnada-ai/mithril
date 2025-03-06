# Program flow

1) Determine context size
    * Tensor data
    * Tensor overhead * n#Tensor
    * Graph overhead
    * Some addttional overhead?
2) Initialize params with context size and some additional parameters
2) Initialize contex with params
4) Create tensors via context and set data
    * We have to set dtype, and shape here.
5) Create graph via context
6) Apply operations to input tensors and obtain result Tensor
    * This step is lazy, actually it creates the computational graph 


### Problems
* Operations are lazy how we are going to do backend operations?
* Shape and dtype must be determined.

### Questions 
* Are we going to create context etc 

### Installation


### Allocation in GGML
1) The context is created with the params. The params contains:
    * The size of the entire graph
    * buffer_mem(if NULL it will allocated) 

2) 
