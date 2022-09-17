#![allow(non_snake_case, unused_macros)]

use rand::prelude::*;
use proconio::input;
use svg::node::{element::{Line, Circle, Group, Title, Rectangle}, Text};

pub trait SetMinMax {
	fn setmin(&mut self, v: Self) -> bool;
	fn setmax(&mut self, v: Self) -> bool;
}
impl<T> SetMinMax for T where T: PartialOrd {
	fn setmin(&mut self, v: T) -> bool {
		*self > v && { *self = v; true }
	}
	fn setmax(&mut self, v: T) -> bool {
		*self < v && { *self = v; true }
	}
}

#[macro_export]
macro_rules! mat {
	($($e:expr),*) => { Vec::from(vec![$($e),*]) };
	($($e:expr,)*) => { Vec::from(vec![$($e),*]) };
	($e:expr; $d:expr) => { Vec::from(vec![$e; $d]) };
	($e:expr; $d:expr $(; $ds:expr)+) => { Vec::from(vec![mat![$e $(; $ds)*]; $d]) };
}

pub const DXY: [(i32, i32); 8] = [
	(1, 0),
	(1, 1),
	(0, 1),
	(-1, 1),
	(-1, 0),
	(-1, -1),
	(0, -1),
	(1, -1)
];

pub type P = (i32, i32);
pub type Output = Vec<[P; 4]>;

#[derive(Clone, Debug)]
pub struct Input {
	pub N: usize,
	pub ps: Vec<P>,
}

impl std::fmt::Display for Input {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		writeln!(f, "{} {}", self.N, self.ps.len())?;
		for p in &self.ps {
			writeln!(f, "{} {}", p.0, p.1)?;
		}
		Ok(())
	}
}

pub fn parse_input(f: &str) -> Input {
	let f = proconio::source::once::OnceSource::from(f);
	input! {
		from f,
		N: usize, M: usize,
		ps: [(i32, i32); M],
	}
	Input { N, ps }
}

fn read<T: Copy + PartialOrd + std::fmt::Display + std::str::FromStr>(token: Option<&str>, lb: T, ub: T) -> Result<T, String> {
	if let Some(v) = token {
		if let Ok(v) = v.parse::<T>() {
			if v < lb || ub < v {
				Err(format!("Out of range: {}", v))
			} else {
				Ok(v)
			}
		} else {
			Err(format!("Parse error: {}", v))
		}
	} else {
		Err("Unexpected EOF".to_owned())
	}
}

pub fn parse_output(input: &Input, f: &str) -> Result<Output, String> {
	let mut out = vec![];
	let mut tokens = f.split_whitespace();
	let M = read(tokens.next(), 0, 1000000000)?;
	for _ in 0..M {
		let mut rect = [(0, 0); 4];
		for i in 0..4 {
			rect[i] = (read(tokens.next(), 0, input.N as i32 - 1)?, read(tokens.next(), 0, input.N as i32 - 1)?);
		}
		out.push(rect);
	}
	Ok(out)
}

#[derive(Clone, Debug)]
pub struct State {
	pub has_point: Vec<Vec<bool>>,
	pub used: Vec<Vec<[bool; 8]>>,
}

impl State {
	pub fn new(input: &Input) -> Self {
		let mut has_point = mat![false; input.N; input.N];
		let used = mat![[false; 8]; input.N; input.N];
		for i in 0..input.ps.len() {
			has_point[input.ps[i].0 as usize][input.ps[i].1 as usize] = true;
		}
		Self {
			has_point,
			used,
		}
	}
	pub fn check_move(&self, rect: [P; 4]) -> String {
		if let Some(i) = (1..4).find(|&i| !self.has_point[rect[i].0 as usize][rect[i].1 as usize]) {
			return format!("({}, {}) does not contain a dot", rect[i].0, rect[i].1);
		} else if self.has_point[rect[0].0 as usize][rect[0].1 as usize] {
			return format!("({}, {}) already contains a dot", rect[0].0, rect[0].1);
		} else {
			let dx01 = rect[1].0 - rect[0].0;
			let dy01 = rect[1].1 - rect[0].1;
			let dx03 = rect[3].0 - rect[0].0;
			let dy03 = rect[3].1 - rect[0].1;
			if dx01 * dx03 + dy01 * dy03 != 0 {
				return "Illegal rectangle".to_owned();
			} else if dx01 != 0 && dy01 != 0 && dx01.abs() != dy01.abs() {
				return "Illegal rectangle".to_owned();
			} else if (rect[1].0 + dx03, rect[1].1 + dy03) != rect[2] {
				return "Illegal rectangle".to_owned();
			} else {
				for i in 0..4 {
					let (mut x, mut y) = rect[i];
					let (tx, ty) = rect[(i + 1) % 4];
					let dx = (tx - x).signum();
					let dy = (ty - y).signum();
					let dir = (0..8).find(|&dir| DXY[dir] == (dx, dy)).unwrap();
					while (x, y) != (tx, ty) {
						if (x, y) != rect[i] && self.has_point[x as usize][y as usize] {
							return format!("There is an obstacle at ({}, {})", x, y);
						}
						if self.used[x as usize][y as usize][dir] {
							return "Overlapped rectangles".to_owned();
						}
						x += dx;
						y += dy;
						if self.used[x as usize][y as usize][dir ^ 4] {
							return "Overlapped rectangles".to_owned();
						}
					}
				}
			}
		}
		String::new()
	}
	pub fn apply_move(&mut self, rect: [P; 4]) {
		self.has_point[rect[0].0 as usize][rect[0].1 as usize] = true;
		for i in 0..4 {
			let (mut x, mut y) = rect[i];
			let (tx, ty) = rect[(i + 1) % 4];
			let dx = (tx - x).signum();
			let dy = (ty - y).signum();
			let dir = (0..8).find(|&dir| DXY[dir] == (dx, dy)).unwrap();
			while (x, y) != (tx, ty) {
				self.used[x as usize][y as usize][dir] = true;
				x += dx;
				y += dy;
				self.used[x as usize][y as usize][dir ^ 4] = true;
			}
		}
	}
}

pub fn weight((x, y): P, N: usize) -> i32 {
	let dx = x - N as i32 / 2;
	let dy = y - N as i32 / 2;
	dx * dx + dy * dy + 1
}

pub fn compute_score(input: &Input, out: &[[P; 4]]) -> (i64, String, State) {
	let mut state = State::new(input);
	for t in 0..out.len() {
		let err = state.check_move(out[t]);
		if err.len() > 0 {
			return (0, format!("{} (turn: {})", err, t), state);
		}
		state.apply_move(out[t]);
	}
	let mut num = 0;
	for &p in &input.ps {
		num += weight(p, input.N);
	}
	for rect in out {
		num += weight(rect[0], input.N);
	}
	let mut den = 0;
	for i in 0..input.N {
		for j in 0..input.N {
			den += weight((i as i32, j as i32), input.N);
		}
	}
	let score = (1e6 * (input.N * input.N) as f64 / input.ps.len() as f64 * num as f64 / den as f64).round() as i64;
	(score, String::new(), state)
}

pub fn gen(seed: u64, n: usize, m: usize) -> Input {
	if seed == 0 && n == 0 && m == 0 {
		return Input {
			N: 33,
			ps: vec![
				(13, 24),
				(14, 24),
				(15, 24),
				(16, 24),
				(17, 24),
				(12, 23),
				(18, 23),
				(11, 22),
				(19, 22),
				(10, 21),
				(20, 21),
				(9, 20),
				(21, 20),
				(8, 19),
				(15, 19),
				(18, 19),
				(22, 19),
				(8, 18),
				(12, 18),
				(15, 18),
				(18, 18),
				(22, 18),
				(8, 17),
				(12, 17),
				(15, 17),
				(18, 17),
				(22, 17),
				(8, 16),
				(12, 16),
				(15, 16),
				(18, 16),
				(22, 16),
				(8, 15),
				(12, 15),
				(15, 15),
				(18, 15),
				(22, 15),
				(9, 14),
				(12, 14),
				(15, 14),
				(18, 14),
				(21, 14),
				(10, 13),
				(12, 13),
				(15, 13),
				(18, 13),
				(20, 13),
				(22, 13),
				(11, 12),
				(12, 12),
				(15, 12),
				(18, 12),
				(19, 12),
				(23, 12),
				(12, 11),
				(15, 11),
				(18, 11),
				(24, 11),
			]
		}
	}
	let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed ^ 3808);
	let mut N = rng.gen_range(15i32, 31i32) as usize * 2 + 1;
	if n > 0 {
		N = n;
	}
	let mut M = rng.gen_range(N as i32, (N * N / 12 + 1).max(N + 1) as i32) as usize;
	if m > 0 {
		M = m;
	}
	let mut ps = vec![];
	for x in N/4..=3*N/4 {
		for y in N/4..=3*N/4 {
			ps.push((x as i32, y as i32));
		}
	}
	ps.shuffle(&mut rng);
	ps.truncate(M);
	Input { N, ps }
}

fn rect(x: i32, y: i32, w: i32, h: i32, fill: &str) -> Rectangle {
	Rectangle::new().set("x", x).set("y", y).set("width", w).set("height", h).set("fill", fill)
}

pub fn vis(input: &Input, out: &[[P; 4]], show_grid: bool) -> (i64, String, String) {
	let D = 800 / input.N;
	let H = input.N * D;
	let W = input.N * D;
	let (score, error, _) = compute_score(input, out);
	let mut doc = svg::Document::new().set("id", "vis").set("viewBox", (0, 0, W, H)).set("width", W).set("height", H);
	doc = doc.add(rect(0, 0, W as i32, H as i32, "white"));
	let mut id = mat![!0; input.N; input.N];
	for p in &input.ps {
		id[p.0 as usize][p.1 as usize] = !1;
	}
	for i in 0..out.len() {
		id[out[i][0].0 as usize][out[i][0].1 as usize] = i;
	}
	if show_grid {
		for x in 0..input.N {
			for y in 0..input.N {
				if id[x][y] == !0 {
					doc = doc.add(Circle::new().set("cx", D / 2 + x * D).set("cy", H - D / 2 - y * D).set("r", 2).set("fill", "gray"));
				}
			}
		}
		
	}
	for rect in out {
		for i in 0..4 {
			let (x1, y1) = rect[i];
			let (x2, y2) = rect[(i + 1) % 4];
			doc = doc.add(Line::new().set("x1", D / 2 + x1 as usize * D).set("y1", H - D / 2 - y1 as usize * D).set("x2", D / 2 + x2 as usize * D).set("y2", H - D / 2 - y2 as usize * D).set("stroke", "black").set("stroke-width", 2));
		}
	}
	for x in 0..input.N {
		for y in 0..input.N {
			let c = if id[x][y] == !0 {
				continue;
			} else if id[x][y] == !1 {
				Circle::new().set("cx", D / 2 + x * D).set("cy", H - D / 2 - y * D).set("r", (D / 4).max(5)).set("fill", "black")
			} else if id[x][y] != out.len() - 1 {
				Circle::new().set("cx", D / 2 + x * D).set("cy", H - D / 2 - y * D).set("r", (D / 4).max(5)).set("fill", "black")
			} else {
				Circle::new().set("cx", D / 2 + x * D).set("cy", H - D / 2 - y * D).set("r", (D / 4).max(5)).set("fill", "red")
			};
			doc = doc.add(c);
		}
	}
	for x in 0..input.N {
		for y in 0..input.N {
			let w = weight((x as i32, y as i32), input.N);
			doc = doc.add(Group::new()
				.add(Title::new().add(Text::new(format!("({}, {})\nw={}", x, y, w)))).add(
				Circle::new().set("cx", D / 2 + x * D).set("cy", H - D / 2 - y * D).set("r", D / 2 - 1).set("fill", "#00000000")
			));
		}
	}
	(score, error, doc.to_string())
}
