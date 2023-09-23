use std::fmt::Debug;

use rand::Rng;

#[derive(Clone)]
pub struct NeuronNetwork {
    pub input_layer_size: u32,
    pub layers: Vec<Layer>,
}

impl NeuronNetwork {
    pub fn create(input_layer_size: u32, layer_builders: Vec<LayerBuilder>) -> NeuronNetwork {
        let mut layers = Vec::new();
        for (i, layer_builder) in layer_builders.iter().enumerate() {
            if i == 0 {
                layers.push(Layer::random(
                    input_layer_size,
                    layer_builder.neuron_count,
                    layer_builder.activation_function.clone(),
                ));
            } else {
                layers.push(Layer::random(
                    layers[i - 1].neuron_count,
                    layer_builder.neuron_count,
                    layer_builder.activation_function.clone(),
                ));
            }
        }
        NeuronNetwork {
            input_layer_size,
            layers,
        }
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
    neuron_count: u32,
    activation_function: ActivationFunction,
}

impl LayerBuilder {
    pub fn new(neuron_count: u32, activation_function: ActivationFunction) -> LayerBuilder {
        Self {
            neuron_count,
            activation_function,
        }
    }
}

#[derive(Clone)]
pub struct Layer {
    pub neuron_count: u32,
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<f32>,
    pub activation_function: ActivationFunction,
}

impl Layer {
    fn random(
        incoming_connections: u32,
        neuron_count: u32,
        activation_function: ActivationFunction,
    ) -> Layer {
        let mut weights: Vec<Vec<f32>> = Vec::new();
        let mut biases: Vec<f32> = Vec::new();

        // Create weight and bias matrix for every neuron
        for _ in 0..neuron_count {
            let mut neuron_weights: Vec<f32> = Vec::new();
            for _ in 0..incoming_connections {
                neuron_weights.push(Layer::random_weight());
            }
            weights.push(neuron_weights);
            biases.push(0.0);
        }
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
        f.debug_struct("Layer")
            .field("weights", &self.weights)
            .field("biases", &self.biases)
            .finish()
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
