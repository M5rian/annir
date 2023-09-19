use gpgpu::{BufOps, DescriptorSet, Framework, GpuBuffer, GpuBufferUsage, Kernel, Program, Shader};

use crate::{Layer, NeuronNetwork};

pub struct Predictor {
    framework: Framework,
}

impl Predictor {
    pub fn default() -> Predictor {
        Self {
            framework: Framework::default(),
        }
    }

    pub fn run_shader(&self) {
        // GPU buffer creation
        let inputs_buffer: GpuBuffer<f32> = GpuBuffer::from_slice(&self.framework, &vec![0.0]);
        let weight_buffer: GpuBuffer<f32> = GpuBuffer::from_slice(&self.framework, &vec![0.0]);
        let bias_buffer: GpuBuffer<f32> = GpuBuffer::from_slice(&self.framework, &vec![0.0]);
        let output_buffer: GpuBuffer<f32> = GpuBuffer::<f32>::with_capacity(&self.framework, 64);

        // Shader load from WGSL source file
        let shader = Shader::from_wgsl_file(&self.framework, "src/shader.wgsl").unwrap();

        // Descriptor set and program creation
        let desc = DescriptorSet::default()
            .bind_buffer(&inputs_buffer, GpuBufferUsage::ReadOnly)
            .bind_buffer(&weight_buffer, GpuBufferUsage::ReadOnly)
            .bind_buffer(&bias_buffer, GpuBufferUsage::ReadOnly)
            .bind_buffer(&output_buffer, GpuBufferUsage::ReadWrite);
        let program = Program::new(&shader, "main").add_descriptor_set(desc); // Entry point

        Kernel::new(&self.framework, program).enqueue(1, 1, 1);

        let output = output_buffer.read_vec_blocking().unwrap(); // Read back C from GPU
        println!("{:?}", output);
    }

    pub fn predict(&self, neuron_network: &NeuronNetwork, inputs: Vec<f32>) -> Vec<f32> {
        if inputs.len() != usize::try_from(neuron_network.input_layer_size).unwrap() {
            panic!(
                "Inputs don't match input neuron count! (Inputs: {}, Neurons: {})",
                inputs.len(),
                neuron_network.input_layer_size
            )
        }

        // Calcuate each layer and parse output back to next layer
        let mut inputs = inputs;
        for layer in &neuron_network.layers {
            let someting = self.predict_layer(&inputs, layer).unwrap();
            inputs = someting;
        }
        inputs
    }

    fn predict_layer(
        &self,
        inputs: &Vec<f32>,
        layer: &Layer,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Cpu data
        let weights = &layer
            .weights
            .iter()
            .flat_map(|inner_vec| inner_vec.iter().cloned())
            .collect::<Vec<f32>>();
        let biases = &layer.biases;

        // GPU buffer creation
        let inputs_buffer = GpuBuffer::from_slice(&self.framework, inputs);
        let weight_buffer = GpuBuffer::from_slice(&self.framework, weights);
        let bias_buffer = GpuBuffer::from_slice(&self.framework, biases);
        let output_buffer =
            GpuBuffer::<f32>::with_capacity(&self.framework, layer.neuron_count as u64);

        // Shader load from WGSL source file
        let shader = Shader::from_wgsl_file(&self.framework, "src/shader.wgsl")?;

        // Descriptor set and program creation
        let desc = DescriptorSet::default()
            .bind_buffer(&inputs_buffer, GpuBufferUsage::ReadOnly)
            .bind_buffer(&weight_buffer, GpuBufferUsage::ReadOnly)
            .bind_buffer(&bias_buffer, GpuBufferUsage::ReadOnly)
            .bind_buffer(&output_buffer, GpuBufferUsage::ReadWrite);
        let program = Program::new(&shader, "main").add_descriptor_set(desc); // Entry point

        let workgroups = (inputs.len() as f32 / 64.0).ceil();
        Kernel::new(&self.framework, program).enqueue(workgroups as u32, 1, 1);

        let output = output_buffer.read_vec_blocking()?; // Read back C from GPU
        Ok(output)
    }
}
