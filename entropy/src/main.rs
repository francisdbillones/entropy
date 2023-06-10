use ndarray::{s, Array2};
use pixel_canvas::{Canvas, Color};
use rand::{rngs::ThreadRng, Rng};
use serde::{Deserialize, Serialize};
use std::iter::zip;
use std::str;
use std::{fs::File, io::BufReader};

#[derive(Debug, Serialize, Deserialize)]
struct Config {
    dims: (usize, usize),
    hotspots: usize,
    sleep_interval_ms: usize,
    heat: f64,
    size_factor: usize,
}

fn main() {
    let config = get_config();

    start_loop(config);
}

#[inline(always)]
fn start_loop(config: Config) {
    let (h, w) = config.dims;

    let mut lagged_board = init_board(&config);
    let mut board = Array2::zeros((h, w));

    let mut rng = rand::thread_rng();

    let canvas = Canvas::new(w * config.size_factor, h * config.size_factor);
    let mut i = 0_usize;

    canvas.render(move |_, image| {
        i += 1;
        println!("{}", i);
        board_time_step(&mut board, &mut lagged_board, &config, &mut rng);

        for (y, row) in image.chunks_mut(w * config.size_factor).enumerate() {
            for (x, pixel) in row.iter_mut().enumerate() {
                let energy = lagged_board[[y / config.size_factor, x / config.size_factor]];
                let rgb = energy_to_rgb(energy, 2.0);
                *pixel = rgb;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(
            config.sleep_interval_ms as u64,
        ));
    });
}

#[inline(always)]
fn board_time_step(
    board: &mut Array2<f64>,
    lagged_board: &mut Array2<f64>,
    config: &Config,
    rng: &mut ThreadRng,
) {
    let (h, w) = config.dims;

    let corner_slices = [
        s![0..2_usize, 0..2_usize], // top left
        s![0..2_usize, w - 2..w],   // top right
        s![h - 2..h, 0..2_usize],   // bottom left
        s![h - 2..h, w - 2..w],     // bottom right
    ];
    let corner_energies: [f64; 4] = [
        lagged_board[[0, 0]],
        lagged_board[[0, w - 1]],
        lagged_board[[h - 1, 0]],
        lagged_board[[h - 1, w - 1]],
    ];

    for (slice, energy) in zip(corner_slices, corner_energies) {
        let mut slice = board.slice_mut(slice);
        slice += &(energy * &probability_mat((2, 2), rng));
    }

    // top and bottom borders
    for j in 1..w - 1 {
        let mut slice = board.slice_mut(s![0..2_usize, j - 1..=j + 1]);
        let energy = lagged_board[[0, j]];
        slice += &(energy * &probability_mat((2, 3), rng));
    }
    for j in 1..w - 1 {
        let mut slice = board.slice_mut(s![h - 2..h, j - 1..=j + 1]);
        let energy = lagged_board[[h - 1, j]];
        slice += &(energy * &probability_mat((2, 3), rng));
    }

    // left to right
    for i in 1..h - 1 {
        // leftmost
        let mut slice = board.slice_mut(s![i - 1..=i + 1, 0..2_usize]);
        let energy = lagged_board[[i, 0]];
        slice += &(energy * &probability_mat((3, 2), rng));

        // in between
        for j in 1..w - 1 {
            let mut slice = board.slice_mut(s![i - 1..=i + 1, j - 1..=j + 1]);
            let energy = lagged_board[[i, j]];
            slice += &(energy * &probability_mat((3, 3), rng));
        }

        // rightmost
        let mut slice = board.slice_mut(s![i - 1..=i + 1, h - 2..h]);
        let energy = lagged_board[[i, w - 1]];
        slice += &(energy * &probability_mat((3, 2), rng));
    }

    lagged_board.clone_from(board);

    board.fill(0.0);
}

#[inline(always)]
fn energy_to_rgb(energy: f64, max_energy: f64) -> Color {
    let (energy, max_energy) = (energy as f64, max_energy as f64);
    let min_hue: f64 = 240.0; // Blue
    let max_hue: f64 = 0.0; // Red
    let normalized_energy = energy / max_energy;
    let hue = min_hue - (normalized_energy * (min_hue - max_hue));
    let (r, g, b) = hsv_to_rgb(hue, 1.0, 1.0);
    Color {
        r: (r * 255.0) as u8,
        g: (g * 255.0) as u8,
        b: (b * 255.0) as u8,
    }
}

#[inline(always)]
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
    let c = v * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());

    let (r, g, b) = if h_prime < 1.0 {
        (c, x, 0.0)
    } else if h_prime < 2.0 {
        (x, c, 0.0)
    } else if h_prime < 3.0 {
        (0.0, c, x)
    } else if h_prime < 4.0 {
        (0.0, x, c)
    } else if h_prime < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    let m = v - c;

    (r + m, g + m, b + m)
}

#[inline(always)]
fn probability_mat((a, b): (usize, usize), rng: &mut ThreadRng) -> Array2<f64> {
    let mut p = Array2::<f64>::zeros((a, b));
    let mut s = 0.0;

    for i in 0..a {
        for j in 0..b {
            p[[i, j]] = rng.gen();
            s += p[[i, j]];
        }
    }

    p /= s;

    p
}

fn init_board(config: &Config) -> Array2<f64> {
    let (h, w) = config.dims;
    let hotspots = config.hotspots;

    let mut board = Array2::<f64>::zeros((h, w));

    let mut rng = rand::thread_rng();

    let mut quota = 0;

    // pad board with negative infinities in its borders
    // board.slice_mut(s![0, 0..w + 2]).fill(-f64::INFINITY);
    // board.slice_mut(s![h + 1, 0..w + 2]).fill(-f64::INFINITY);
    // board.slice_mut(s![0..h + 2, 0]).fill(-f64::INFINITY);
    // board.slice_mut(s![0..h + 2, w + 1]).fill(-f64::INFINITY);

    while quota != hotspots {
        let rx = rng.gen_range(0..w);
        let ry = rng.gen_range(0..h);

        if board[[ry, rx]] != 0.0 {
            continue;
        }

        let (h, w, hotspots) = (h as f64, w as f64, hotspots as f64);
        board[[ry, rx]] = (h * w * w / h) / hotspots;
        quota += 1;
    }

    board
}

fn get_config() -> Config {
    let path = "config.json";
    let file = File::open(path).expect("Couldn't find config.json");
    let reader = BufReader::new(file);

    let config: Config = serde_json::from_reader(reader).expect("Couldn't parse json");

    config
}
