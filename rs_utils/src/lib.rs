use std::collections::BinaryHeap;

use pyo3::prelude::*;
use rayon::prelude::*;
#[derive(Copy, Clone, PartialEq)]
struct SegmentPoint {
    x: f32,
    y: f32,
}
#[derive(Copy, Clone, PartialEq)]
struct SegmentLine {
    index: usize,
    point1: SegmentPoint,
    point2: SegmentPoint,
    point_count: u16,
    length: f32,
}

impl Eq for SegmentPoint {}
impl Eq for SegmentLine {}

impl Ord for SegmentLine {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let weight_self = self.length / f32::from(self.point_count);
        let weight_other = other.length / f32::from(other.point_count);
        if weight_self > weight_other {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Less
        }
    }
}

impl PartialOrd for SegmentLine {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl SegmentLine {
    fn add_point(&mut self) {
        self.point_count += 1;
    }

    fn new(index: usize, point1: SegmentPoint, point2: SegmentPoint) -> Self {
        Self {
            index: index,
            point1,
            point2,
            point_count: 1,
            length: point1.distance(&point2),
        }
    }
}

impl SegmentPoint {
    fn distance(&self, other: &SegmentPoint) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

fn resample_segment(segments: &Vec<f32>, n: usize) -> Vec<f32> {
    let mut heap: BinaryHeap<SegmentLine> = BinaryHeap::new();

    // Init initial queue
    let mut total_point_count = segments.len() - 1;
    for (index, window) in segments.windows(4).step_by(2).enumerate() {
        heap.push(SegmentLine::new(
            index,
            SegmentPoint {
                x: window[0],
                y: window[1],
            },
            SegmentPoint {
                x: window[2],
                y: window[3],
            },
        ));
    }

    while total_point_count < n {
        let mut line = heap.pop().unwrap();
        line.add_point();
        heap.push(line);
        total_point_count += 1;
    }
    // println!("Total add point: {}",time);

    let mut sorted: Vec<SegmentLine> = heap
        .into_sorted_vec()
        .par_iter()
        .map(|&a| a)
        .rev()
        .take(n - 1)
        .collect();

    sorted.par_sort_by(|a, b| a.index.cmp(&b.index));

    let mut points: Vec<f32> = Vec::new();

    let mut length = sorted.len();
    for segment in sorted.iter() {
        // println!("{:?}", points);
        length -= 1;
        points.push(segment.point1.x);
        points.push(segment.point1.y);

        for i in 0..segment.point_count {
            let time;
            if segment.point_count > 1 {
                time = f32::from(i) / f32::from(segment.point_count - 1);
            } else {
                if n - 1 - length > (points.len() / 2) {
                    time = 0.5;
                } else {
                    continue;
                }
            }
            points.push(segment.point1.x + time * (segment.point2.x - segment.point1.x));
            points.push(segment.point1.y + time * (segment.point2.y - segment.point1.y));
        }
    }
    points.append(segments.last_chunk::<2>().unwrap().to_vec().as_mut());
    points
}

#[pyfunction]
fn resample_segments(segments: Vec<Vec<f32>>, n: usize) -> PyResult<Vec<Vec<f32>>> {
    let segments: Vec<Vec<f32>> = segments
        .par_iter()
        .map(|segment| resample_segment(segment, n))
        .collect();
    Ok(segments)
}

/// A Python module implemented in Rust.
#[pymodule]
fn rs_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(resample_segments, m)?)?;
    Ok(())
}
