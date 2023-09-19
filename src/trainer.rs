use rand::seq::SliceRandom;

use crate::{predictor::Predictor, NeuronNetwork};

pub struct Trainer {
    pub training_data: Vec<TrainingData>,
    pub batch_size: u32,
}

#[derive(Clone)]
pub struct TrainingData {
    pub inputs: Vec<f32>,
    pub outputs: Vec<f32>,
}

impl Trainer {
    pub fn train(&self, predictor: &Predictor, neuron_network: &mut NeuronNetwork, num: i32) {
        for _index in 0..num {
            self.learn(predictor, neuron_network);
        }
    }

    fn generate_batch(&self) -> Vec<TrainingData> {
        let batch_size = self.batch_size.try_into().unwrap();
        self.training_data
            .choose_multiple(&mut rand::thread_rng(), batch_size)
            .cloned()
            .collect::<Vec<TrainingData>>()
    }

    fn learn(&self, predictor: &Predictor, neuron_network: &mut NeuronNetwork) {
        // Constants for learning
        let distance = 0.00001;
        let learn_rate = 0.01;

        // Generate a training batch
        let batch = self.generate_batch();

        // Calculate the initial cost to compare later to
        let original_cost = self.average_cost(predictor, neuron_network, batch.clone());
        println!("Original Cost: {}", original_cost);

        let cloned_network = neuron_network.clone();

        // Iterate through each layer
        for (layer_index, layer) in cloned_network.clone().layers.iter_mut().enumerate() {
            println!("layer: {}", layer_index);
            // Iterate through weights
            for (weight_row_index, weight_row) in layer.weights.iter_mut().enumerate() {
                for (weight_index, weight) in weight_row.iter_mut().enumerate() {
                    println!(
                        "Weight Row Index: {}, Weight Index: {}",
                        weight_row_index, weight_index
                    );
                    // Adjust the weight and calculate the new cost
                    *weight += distance;
                    let new_cost = self.average_cost(predictor, &cloned_network, batch.clone());
                    let slope = (new_cost - original_cost) / distance;
                    *weight += -distance + (slope * learn_rate);
                }
            }

            // Iterate through biases
            for bias in &mut layer.biases {
                // Adjust the bias and calculate the new cost
                *bias += distance;
                let new_cost = self.average_cost(predictor, &cloned_network, batch.clone());
                let slope = (new_cost - original_cost) / distance;
                *bias += -distance + (slope * learn_rate);
            }
        }

        *neuron_network = cloned_network;
    }

    fn average_cost(
        &self,
        predictor: &Predictor,
        neuron_network: &NeuronNetwork,
        training_data: Vec<TrainingData>,
    ) -> f32 {
        let training_data_size = training_data.len() as f32; // TODO casting is bad ig
        let mut total_cost = 0.0;
        for dataset in training_data {
            total_cost += self.cost(predictor, neuron_network, dataset);
        }
        total_cost / training_data_size
    }

    fn cost(
        &self,
        predictor: &Predictor,
        neuron_network: &NeuronNetwork,
        training_data: TrainingData,
    ) -> f32 {
        let prediction = predictor.predict(neuron_network, training_data.inputs);
        let mut loss = 0.0;
        for (i, expected_output) in training_data.outputs.iter().enumerate() {
            loss += self.calculate_loss(*expected_output, prediction[i]);
        }
        loss
    }

    fn calculate_loss(&self, expected_output: f32, output: f32) -> f32 {
        let error = output - expected_output;
        error * error
    }
}
