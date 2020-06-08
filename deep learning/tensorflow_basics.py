import tensorflow as tf

#%% Simple NN

class myDenseLayer(tf.keras.layers.Layer):
    
    def __init__(self, input_dim, output_dim):
        super(myDenseLayer, self).__init__()
        # Initialize weights and bias
        self.W = self.add_weight([input_dim, output_dim])
        self.b = self.add_weight([1, output_dim])
        
    def call(self, inputs):
        # Forward propagate
        z = tf.matmul(inputs, self.W) + self.b
        output = tf.math.sigmoid(z)
        return output

# Which is equivalent to:
# layer = tf.keras.layers.Dense(units = 2)  

    
weights = tf.Variable([tf.random.normal()])

while True:
    with tf.GradientTape() as g:
        loss = compute_loss(weights)
        gradient = g.gradient(loss, weights) 
    weights = weights - lr * gradient
    
#%% Putting all together
 
# Define architecture of a model       
model = tf.keras.Sequential([
        tf.keras.layers.Dense(n1),
        tf.keras.layers.Dense(n2),
        tf.keras.layers.Dense(n3),
        tf.keras.layers.Dense(2),
        ])
    
# Define optimizer    
optimizer = tf.keras.optimizer.SGD()
    
while True:
    prediction = model(X) # Forward pass through the model
    with tf.GradientTape() as tape:
        # Compute the loss
        loss = compute_loss(y, prediction)
    # Update the weights using the gradient
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
# Keep looping: it will converge
    