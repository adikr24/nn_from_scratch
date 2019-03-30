# nn_from_scratch
neural network from scratch with back propagation in n layers


the first half of the code walks through the math of the backpropagation and shows how it's performed in different layers (all the way from 5th
to the first) and then shows how the weights are updated 

Now once the math is clear, I tried to implement OOPs concept to create n layer neural network which can take arrays as inputs and take in
user defined hidden_layer with different sizes and can perform sigmoid activation function and pass them through series of hidden_layer and
present the output, the backpropagaiton function when called finds the derivate of the cost function w.r.t to the weights in different layers
and then yields the delta change which when multiplied with learning_parameter or learning rate is reduced from the original weights (to create
the update weights) and finally the updated weights will be further used to calculate the Y_pred. By running the code multiple times we can
see that the weights have changed and the cost function has reduced.

different cost functions can be used and momentum can be added but for the sake of simplicity this was avoided.

