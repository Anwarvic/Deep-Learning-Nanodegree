# Implementing backpropagation

Now we've seen that the error term for the output layer is

\delta_k = (y_k - \hat y_k) f'(a_k)δk=(yk−y^k)f′(ak)

and the error term for the hidden layer is

[![img](https://d17h27t6h515a5.cloudfront.net/topher/2017/January/588bc453_hidden-errors/hidden-errors.gif)](https://classroom.udacity.com/nanodegrees/nd101-ent/parts/7de57e8c-12ed-4d7a-b8ea-15db419f6a58/modules/9f27732b-a272-4d8d-8cf3-28159ebc7200/lessons/85c95c4c-4b3a-42d8-983b-9f760fe38055/concepts/b2bbdc9a-9f48-4735-b408-71cf67f5b000#)

For now we'll only consider a simple network with one hidden layer and one output unit. Here's the general algorithm for updating the weights with backpropagation:

- Set the weight steps for each layer to zero
  - The input to hidden weights \Delta w_{ij} = 0Δwij=0
  - The hidden to output weights \Delta W_j = 0ΔWj=0
- For each record in the training data:
  - Make a forward pass through the network, calculating the output \hat yy^
  - Calculate the error gradient in the output unit, \delta^o = (y - \hat y) f'(z)δo=(y−y^)f′(z) where z = \sum_j W_j a_jz=∑jWjaj, the input to the output unit.
  - Propagate the errors to the hidden layer \delta^h_j = \delta^o W_j f'(h_j)δjh=δoWjf′(hj)
  - Update the weight steps:
    - \Delta W_j = \Delta W_j + \delta^o a_jΔWj=ΔWj+δoaj
    - \Delta w_{ij} = \Delta w_{ij} + \delta^h_j a_iΔwij=Δwij+δjhai
- Update the weights, where \etaη is the learning rate and mm is the number of records:
  - W_j = W_j + \eta \Delta W_j / mWj=Wj+ηΔWj/m
  - w_{ij} = w_{ij} + \eta \Delta w_{ij} / mwij=wij+ηΔwij/m
- Repeat for ee epochs.

## Backpropagation exercise

Now you're going to implement the backprop algorithm for a network trained on the graduate school admission data. You should have everything you need from the previous exercises to complete this one.

Your goals here:

- Implement the forward pass.
- Implement the backpropagation algorithm.
- Update the weights.