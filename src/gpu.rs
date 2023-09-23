use std::vec;

use gpgpu::{BufOps, DescriptorSet, Framework, GpuBuffer, GpuBufferUsage, Kernel, Program, Shader};

use crate::network::Layer;

pub struct Gpu {
    framework: Framework,
}

impl Gpu {
    pub fn default() -> Gpu {
        Self { framework: Framework::default() }
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

    pub fn predict_layer(&self, inputs: &Vec<f32>, layer: &Layer) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Cpu data
        let weights = &layer.weights.to_data();
        let biases = &layer.biases;

        // GPU buffer creation
        let inputs_buffer = GpuBuffer::from_slice(&self.framework, inputs);
        let weight_buffer = GpuBuffer::from_slice(&self.framework, weights);
        let bias_buffer = GpuBuffer::from_slice(&self.framework, biases);
        let output_buffer = GpuBuffer::<f32>::with_capacity(&self.framework, layer.neuron_count as u64);

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
        let mut activation_applied = Vec::with_capacity(output.len());
        for i in 0..output.len() {
            activation_applied.push(layer.activation_function.apply(&output, i))
        }
        Ok(activation_applied)
    }
}
