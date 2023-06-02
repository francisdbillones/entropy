use pixel_canvas::{Canvas, Color};
use rand::{rngs::ThreadRng, Rng};
use serde::{Deserialize, Serialize};
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
    let mut board = vec![vec![0; w]; h];

    let mut rng = rand::thread_rng();

    let canvas = Canvas::new(w * config.size_factor, h * config.size_factor);

    canvas.render(move |_, image| {
        board_time_step(&mut board, &mut lagged_board, &config, &mut rng);

        for (y, row) in image.chunks_mut(w * config.size_factor).enumerate() {
            for (x, pixel) in row.iter_mut().enumerate() {
                let energy = lagged_board[y / config.size_factor][x / config.size_factor];
                let rgb = energy_to_rgb(energy, 2);
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
    board: &mut [Vec<usize>],
    lagged_board: &mut [Vec<usize>],
    config: &Config,
    rng: &mut ThreadRng,
) {
    let (h, w) = config.dims;
    for (i, row) in lagged_board.iter().enumerate() {
        for (j, &energy) in row.iter().enumerate() {
            random_steps(board, (i, j), energy, config, rng);
        }
    }

    for i in 0..h {
        for j in 0..w {
            lagged_board[i][j] = board[i][j];
        }
    }

    for i in 0..h {
        for j in 0..w {
            board[i][j] = 0;
        }
    }
}

#[inline(always)]
fn random_steps(
    board: &mut [Vec<usize>],
    (i, j): (usize, usize),
    energy: usize,
    config: &Config,
    rng: &mut ThreadRng,
) {
    if rng.gen::<f64>() > config.heat {
        return;
    }

    let (count, neighbours) = iterate_neighbours((i, j), config);

    let percentages = gen_percentages(count, rng);

    let mut s = 0;
    for (cur_percentage, (ci, cj)) in neighbours.iter().take(count).enumerate() {
        let dispersed_energy = (energy as f64 * percentages[cur_percentage]).floor() as usize;
        board[*ci][*cj] += dispersed_energy;
        s += dispersed_energy;
    }
    let (ri, rj) = neighbours[rng.gen_range(0..count)];
    board[ri][rj] += energy - s;
}

#[inline(always)]
fn iterate_neighbours((i, j): (usize, usize), config: &Config) -> (usize, [(usize, usize); 9]) {
    let (h, w) = config.dims;
    let (i, j) = (i as isize, j as isize);

    let mut neighbours = [(0_usize, 0_usize); 9];
    let mut count = 0;

    for i in i - 1..=i + 1 {
        if i < 0 || i >= h as isize {
            continue;
        }
        for j in j - 1..=j + 1 {
            if j < 0 || j >= w as isize {
                continue;
            }
            neighbours[{
                count += 1;
                count - 1
            }] = (i as usize, j as usize);
        }
    }

    (count, neighbours)
}

#[inline(always)]
fn energy_to_rgb(energy: usize, max_energy: usize) -> Color {
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
fn gen_percentages(n: usize, rng: &mut ThreadRng) -> [f64; 9] {
    assert!(n <= 9);

    let mut p: [f64; 9] = [0.0; 9];

    let mut s = 0.0;
    for v in p.iter_mut().take(n) {
        let r = rng.gen();
        *v = r;
        s += r;
    }

    for v in p.iter_mut().take(n) {
        *v /= s;
    }

    p
}

fn init_board(config: &Config) -> Vec<Vec<usize>> {
    let (h, w) = config.dims;
    let hotspots = config.hotspots;

    let mut board = vec![vec![0_usize; w]; h];

    let mut rng = rand::thread_rng();

    let mut quota = 0;

    loop {
        if quota == hotspots {
            break;
        }

        let rx = rng.gen_range(0..w);
        let ry = rng.gen_range(0..h);

        if board[ry][rx] != 0 {
            continue;
        }

        board[ry][rx] = (h * w * w / h) / hotspots;
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
