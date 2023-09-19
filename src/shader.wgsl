struct Vector {
    data: array<f32>;
};

[[group(0), binding(0)]] var<storage, read> inputs: Vector;
[[group(0), binding(1)]] var<storage, read> weights: Vector;
[[group(0), binding(2)]] var<storage, read> biases: Vector;
[[group(0), binding(3)]] var<storage, read_write> output: Vector;

// Workgroup size of 64 is recommended on AMD gpus.
// Nividia recommends a multiplier of 32.
[[stage(compute), workgroup_size(4,4,4)]]
fn main(
    [[builtin(local_invocation_index)]] local_invocation_index: u32,
) {
    // Only run shader if shader didn't exceed the inputs length limit
    if (arrayLength(&inputs.data) < local_invocation_index) {
        let weights_size = arrayLength(&inputs.data); // Total length of weight matrix
        let weights_columns = weights_size / arrayLength(&inputs.data); // Amount of columns in weight matrix

        var result: f32 = 0.0;
        for (var i: u32 = 0u; i < weights_columns; i = i + 1u) {
            let product = inputs.data[i] * weights.data[weights_columns * local_invocation_index + i];
            result = result + product;
        }

        output.data[local_invocation_index] = result + biases.data[local_invocation_index];
    }
}