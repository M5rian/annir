use std::vec;

use std::ops::{Index, IndexMut};

#[derive(Clone)]
pub struct Matrix {
    rows: usize,
    columns: usize,
    data: Vec<Vec<f32>>,
}

impl Matrix {
    pub fn new(rows: usize, columns: usize) -> Self {
        let data = vec![vec![0.0; columns]; rows];
        Self {
            rows,
            columns,
            data,
        }
    }

    pub fn to_data(&self) -> Vec<f32> {
        let mut vector = Vec::new();
        for i in 0..self.data.len() {
            for j in 0..self.data[i].len() {
                vector.push(self.data[i][j]);
            }
        }
        vector
    }

    pub fn set(&mut self, i: usize, j: usize, value: f32) {
        self.data[i][j] = value;
    }
}

impl Index<usize> for Matrix {
    type Output = Vec<f32>;

    fn index(&self, i: usize) -> &Self::Output {
        &self.data[i]
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.data[i]
    }
}
