use crate::{gpu::Gpu, matrix::Matrix};
use rand::Rng;
use std::fmt::Debug;

#[derive(Clone)]
pub struct NeuronNetwork {
    pub input_layer_size: usize,
    pub layers: Vec<Layer>,
}

impl NeuronNetwork {
    pub fn create(input_layer_size: usize, layer_builders: Vec<LayerBuilder>) -> NeuronNetwork {
        let mut layers = Vec::new();
        let mut prev_neuron_count = input_layer_size;

        for layer_builder in &layer_builders {
            let neuron_count = layer_builder.neuron_count;
            let activation_function = layer_builder.activation_function.clone();
            layers.push(Layer::random(prev_neuron_count, neuron_count, activation_function.clone()));
            prev_neuron_count = neuron_count;
        }

        NeuronNetwork { input_layer_size, layers }
    }

    pub fn feed_forward(&self, inputs: Vec<f32>) -> Vec<f32> {
        if inputs.len() != self.input_layer_size {
            panic!("Inputs don't match input neuron count! (Inputs: {}, Neurons: {})", inputs.len(), &self.input_layer_size);
        }

        let gpu = Gpu::default();
        // Calcuate each layer and parse output back to next layer
        let mut inputs = inputs;
        for layer in &self.layers {
            let someting = gpu.predict_layer(&inputs, layer).unwrap();
            inputs = someting;
        }
        inputs
    }
}

impl Debug for NeuronNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeuronNetwork")
            .field("input_layer_size", &self.input_layer_size)
            .field("layer_count", &self.layers.len())
            .finish()
    }
}

pub struct LayerBuilder {
    neuron_count: usize,
    activation_function: ActivationFunction,
}

impl LayerBuilder {
    pub fn new(neuron_count: usize, activation_function: ActivationFunction) -> LayerBuilder {
        Self {
            neuron_count,
            activation_function,
        }
    }
}

#[derive(Clone)]
pub struct Layer {
    pub neuron_count: usize,
    pub weights: Matrix,
    pub biases: Vec<f32>,
    pub activation_function: ActivationFunction,
}

impl Layer {
    fn random(incoming_connections: usize, neuron_count: usize, activation_function: ActivationFunction) -> Layer {
        let mut weights = Matrix::new(neuron_count, incoming_connections);
        for i in 0..neuron_count as usize {
            for j in 0..incoming_connections as usize {
                weights[i][j] = Layer::random_weight()
            }
        }

        let mut biases: Vec<f32> = vec![0.0; neuron_count];

        Layer {
            neuron_count,
            weights,
            biases,
            activation_function,
        }
    }

    /// Generates random weight between 0 and 1.
    fn random_weight() -> f32 {
        rand::thread_rng().gen()
    }
}

impl Debug for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Layer").field("weights", &self.weights.to_data()).field("biases", &self.biases).finish()
    }
}

#[derive(Clone)]
pub enum ActivationFunction {
    ReLU,
    SoftMax,
    Linear,
}

impl ActivationFunction {
    pub fn apply(&self, vector: &Vec<f32>, index: usize) -> f32 {
        match self {
            ActivationFunction::ReLU => vector[index].max(0.0),
            ActivationFunction::Linear => vector[index],
            ActivationFunction::SoftMax => {
                let sum: f32 = vector.iter().sum();
                vector[index] / sum
            }
        }
    }
}
