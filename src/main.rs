use std::{
    fmt::Debug,
    fs::File,
    io::{BufReader, Read},
    time::Instant,
    vec,
};

use rand::Rng;

use crate::{
    predictor::Predictor,
    trainer::{Trainer, TrainingData},
};

mod predictor;
mod trainer;

fn main() {
    let training_data = load_training_data();
    println!("Inputs: {}", training_data[0].inputs.len());
    println!("Outputs: {}", training_data[0].outputs.len());

    let mut neuron_network = NeuronNetwork::from_layer_sizes(vec![28 * 28, 50, 10]);
    println!("{:?}", neuron_network);

    let predictor = Predictor::default();

    let now = Instant::now();
    let prediction = predictor.predict(&neuron_network, training_data[0].inputs.clone());
    println!("Prediction took {}ms", now.elapsed().as_millis());
    println!("{:?}", prediction);

    let trainer = Trainer {
        training_data,
        batch_size: 5,
    };
    trainer.train(&predictor, &mut neuron_network, 6);
}

fn load_training_data() -> Vec<TrainingData> {
    // Open the label file
    let labels_file =
        File::open("./mnist-database/train-labels").expect("Failed to read label file");
    let mut labels_reader = BufReader::new(labels_file);

    // Read the magic number from the label file
    let mut magic_number_buffer = [0; 4];
    labels_reader.read_exact(&mut magic_number_buffer).unwrap();
    let magic_number = i32::from_be_bytes(magic_number_buffer);
    println!("Magic number (labels): {}", magic_number);

    // Read the number of items from the label file
    let mut number_of_items_buffer = [0; 4];
    labels_reader
        .read_exact(&mut number_of_items_buffer)
        .unwrap();
    let number_of_items = i32::from_be_bytes(number_of_items_buffer);
    println!("Number of items: {}", number_of_items);

    // Read label data into a vector
    let mut labels = Vec::new();
    labels_reader.read_to_end(&mut labels).unwrap();

    // Format labels into one-hot encoding
    let mut formatted_labels = Vec::new();
    for value in &labels {
        let mut output_for_value = vec![0.0; 10];
        output_for_value[*value as usize] = 1.0;
        formatted_labels.push(output_for_value);
    }

    // Print example labels
    println!("1st Label: {}", labels[0]);
    println!("2nd Label: {}", labels[1]);
    println!("10th Label: {}", labels[9]);
    println!("25th Label: {}", labels[24]);

    // Open the images file
    let images_file =
        File::open("./mnist-database/train-images").expect("Failed to read images file");
    let mut images_reader = BufReader::new(images_file);

    // Read the magic number from the images file
    let mut magic_number_images_buffer = [0; 4];
    images_reader
        .read_exact(&mut magic_number_images_buffer)
        .unwrap();
    let magic_number_images = i32::from_be_bytes(magic_number_images_buffer);
    println!("Magic number (images): {}", magic_number_images);

    // Read the number of images, rows, and columns from the images file
    let mut number_of_images_buffer = [0; 4];
    images_reader
        .read_exact(&mut number_of_images_buffer)
        .unwrap();
    let number_of_images = i32::from_be_bytes(number_of_images_buffer);
    println!("Number of images: {}", number_of_images);

    let mut rows_buffer = [0; 4];
    images_reader.read_exact(&mut rows_buffer).unwrap();
    let rows = i32::from_be_bytes(rows_buffer);
    println!("Rows: {}", rows);

    let mut columns_buffer = [0; 4];
    images_reader.read_exact(&mut columns_buffer).unwrap();
    let columns = i32::from_be_bytes(columns_buffer);
    println!("Columns: {}", columns);

    // Read image pixel data into a vector
    let mut pixels = Vec::new();
    images_reader.read_to_end(&mut pixels).unwrap();

    // Process pixel data into images
    let mut images = Vec::new();
    let mut current_image = Vec::new();
    let pixels_in_image = (rows * columns) as usize;
    println!("{} pixels in image", pixels_in_image);
    for pixel in pixels {
        let decimal_pixel = pixel as f32 / 255.0;
        current_image.push(decimal_pixel);

        if current_image.len() == pixels_in_image {
            images.push(current_image.clone());
            current_image.clear();
        }
    }

    // Create the training data set
    let mut training_set = Vec::new();
    for i in 0..number_of_images as usize {
        let training_data = TrainingData {
            inputs: images[i].clone(),
            outputs: formatted_labels[i].clone(),
        };
        training_set.push(training_data);
    }
    training_set
}

#[derive(Clone)]
pub struct NeuronNetwork {
    input_layer_size: u32,
    layers: Vec<Layer>,
}

impl NeuronNetwork {
    fn from_layer_sizes(mut layer_sizes: Vec<u32>) -> NeuronNetwork {
        let mut layers: Vec<Layer> = Vec::new();
        for (layer_index, layer_size) in layer_sizes.iter().enumerate() {
            if layer_index == 0 {
                continue;
            }

            let previous_layer_size = layer_sizes[layer_index - 1];
            let layer = Layer::random(previous_layer_size, *layer_size);
            layers.push(layer);
        }

        let input_layer_size = layer_sizes.remove(0);

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

#[derive(Clone)]
struct Layer {
    neuron_count: u32,
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
}

impl Layer {
    fn random(incoming_connections: u32, neuron_count: u32) -> Layer {
        let mut weights: Vec<Vec<f32>> = Vec::new();
        let mut biases: Vec<f32> = Vec::new();

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
