use std::{mem::swap, time::Instant};

use bitboard::Board;
#[allow(unused_imports)]
use proconio::*;
#[allow(unused_imports)]
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use vector::DIR_COUNT;

use crate::vector::{rot_c, rot_cc, Vec2};

#[allow(unused_macros)]
macro_rules! chmin {
    ($base:expr, $($cmps:expr),+ $(,)*) => {{
        let cmp_min = min!($($cmps),+);
        if $base > cmp_min {
            $base = cmp_min;
            true
        } else {
            false
        }
    }};
}

#[allow(unused_macros)]
macro_rules! chmax {
    ($base:expr, $($cmps:expr),+ $(,)*) => {{
        let cmp_max = max!($($cmps),+);
        if $base < cmp_max {
            $base = cmp_max;
            true
        } else {
            false
        }
    }};
}

#[allow(unused_macros)]
macro_rules! min {
    ($a:expr $(,)*) => {{
        $a
    }};
    ($a:expr, $b:expr $(,)*) => {{
        std::cmp::min($a, $b)
    }};
    ($a:expr, $($rest:expr),+ $(,)*) => {{
        std::cmp::min($a, min!($($rest),+))
    }};
}

#[allow(unused_macros)]
macro_rules! max {
    ($a:expr $(,)*) => {{
        $a
    }};
    ($a:expr, $b:expr $(,)*) => {{
        std::cmp::max($a, $b)
    }};
    ($a:expr, $($rest:expr),+ $(,)*) => {{
        std::cmp::max($a, max!($($rest),+))
    }};
}

#[allow(unused_macros)]
macro_rules! mat {
    ($e:expr; $d:expr) => { vec![$e; $d] };
    ($e:expr; $d:expr $(; $ds:expr)+) => { vec![mat![$e $(; $ds)*]; $d] };
}

#[allow(unused_macros)]
macro_rules! skip_none {
    ($res:expr) => {
        if let Some(v) = $res {
            v
        } else {
            continue;
        }
    };
}

mod bitboard {
    use crate::vector::{Vec2, DIR_COUNT, UNITS};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    struct Bitset {
        v: u64,
    }

    impl Bitset {
        #[allow(dead_code)]
        const fn new(v: u64) -> Self {
            Self { v }
        }

        fn at(&self, i: u32) -> bool {
            ((self.v >> i) & 1) > 0
        }

        fn set(&mut self, i: u32) {
            debug_assert!(((self.v >> i) & 1) == 0);
            self.v ^= 1 << i;
        }

        fn unset(&mut self, i: u32) {
            debug_assert!(((self.v >> i) & 1) > 0);
            self.v ^= 1 << i;
        }

        fn find_next(&self, begin: u32) -> Option<u32> {
            let v = self.v >> begin;
            if v == 0 {
                None
            } else {
                let tz = v.trailing_zeros();
                Some(begin + tz)
            }
        }

        fn contains_range(&self, begin: u32, end: u32) -> bool {
            debug_assert!(begin <= end);
            (self.v & Self::get_range_mask(begin, end)) > 0
        }

        fn set_range(&mut self, begin: u32, end: u32) {
            debug_assert!(!self.contains_range(begin, end));
            self.v ^= Self::get_range_mask(begin, end);
        }

        fn unset_range(&mut self, begin: u32, end: u32) {
            let mask = Self::get_range_mask(begin, end);
            debug_assert!((self.v & mask) == mask);
            self.v ^= mask;
        }

        fn get_range_popcnt(&self, begin: u32, end: u32) -> u32 {
            let mask = Self::get_range_mask(begin, end);
            (self.v & mask).count_ones()
        }

        fn get_range_mask(begin: u32, end: u32) -> u64 {
            debug_assert!(end >= begin);
            ((1 << end) - 1) ^ ((1 << begin) - 1)
        }
    }

    impl std::fmt::Display for Bitset {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:b}", self.v)
        }
    }

    #[derive(Debug, Clone)]
    pub struct Board {
        n: usize,
        points: [Vec<Bitset>; DIR_COUNT],
        edges: [Vec<Bitset>; DIR_COUNT / 2],
    }

    impl Board {
        pub fn init(n: usize, init_points: &[Vec2]) -> Self {
            let mut points = [
                vec![Bitset::default(); n],
                vec![Bitset::default(); 2 * n - 1],
                vec![Bitset::default(); n],
                vec![Bitset::default(); 2 * n - 1],
                vec![Bitset::default(); n],
                vec![Bitset::default(); 2 * n - 1],
                vec![Bitset::default(); n],
                vec![Bitset::default(); 2 * n - 1],
            ];
            let edges = [
                vec![Bitset::default(); n],
                vec![Bitset::default(); 2 * n - 1],
                vec![Bitset::default(); n],
                vec![Bitset::default(); 2 * n - 1],
            ];

            for p in init_points.iter() {
                for (dir, points) in points.iter_mut().enumerate() {
                    let p = p.rot(dir, n);
                    points[p.y as usize].set(p.x as u32);
                }
            }

            Self { n, points, edges }
        }

        pub fn find_next(&self, v: Vec2, dir: usize) -> Option<Vec2> {
            unsafe {
                let v_rot = v.rot(dir, self.n);
                let next = self
                    .points
                    .get_unchecked(dir)
                    .get_unchecked(v_rot.y as usize)
                    .find_next(v_rot.x as u32 + 1);

                if let Some(next) = next {
                    let unit = *UNITS.get_unchecked(dir);
                    let d = next as i32 - v_rot.x;

                    let (x1, x2, y, dir) = if (dir & 4) == 0 {
                        (v_rot.x, v_rot.x + d, v_rot.y, dir)
                    } else {
                        let dir = dir & 3;
                        let v_rot = v.rot(dir, self.n);
                        let x1 = v_rot.x - d;
                        let x2 = v_rot.x;
                        (x1, x2, v_rot.y, dir)
                    };

                    let has_edge = self
                        .edges
                        .get_unchecked(dir)
                        .get_unchecked(y as usize)
                        .contains_range(x1 as u32, x2 as u32);

                    if has_edge {
                        None
                    } else {
                        let next = v + unit * d;
                        Some(next)
                    }
                } else {
                    None
                }
            }
        }

        pub fn is_occupied(&self, v1: Vec2) -> bool {
            self.points[0][v1.y as usize].at(v1.x as u32)
        }

        pub fn can_connect(&self, v1: Vec2, v2: Vec2) -> bool {
            let (dir, y, x1, x2) = self.get_rot4(v1, v2);
            let has_point = self.points[dir][y].contains_range(x1 + 1, x2);
            let has_edge = self.edges[dir][y].contains_range(x1, x2);
            !has_point && !has_edge
        }

        pub fn add_point(&mut self, v: Vec2) {
            for dir in 0..DIR_COUNT {
                let v = v.rot(dir, self.n);
                self.points[dir][v.y as usize].set(v.x as u32);
            }
        }

        pub fn remove_point(&mut self, v: Vec2) {
            for dir in 0..DIR_COUNT {
                let v = v.rot(dir, self.n);
                self.points[dir][v.y as usize].unset(v.x as u32);
            }
        }

        pub fn get_range_popcnt(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> usize {
            let mut count = 0;

            for y in y0..y1 {
                count += self.points[0][y].get_range_popcnt(x0 as u32, x1 as u32);
            }

            count as usize
        }

        pub fn connect_parallel(&mut self, v0: Vec2, width: i32, height: i32, dir: usize) {
            let v0 = v0.rot(dir, self.n);
            let y0 = v0.y as usize;
            let x0 = v0.x as u32;
            let y1 = (y0 as i32 + height) as usize;
            let x1 = x0 + width as u32;
            let edges = &mut self.edges[dir];
            edges[y0].set_range(x0, x1);
            edges[y1].set_range(x0, x1);
        }

        pub fn disconnect_parallel(&mut self, v0: Vec2, width: i32, height: i32, dir: usize) {
            let v0 = v0.rot(dir, self.n);
            let y0 = v0.y as usize;
            let x0 = v0.x as u32;
            let y1 = (y0 as i32 + height) as usize;
            let x1 = x0 + width as u32;
            let edges = &mut self.edges[dir];
            edges[y0].unset_range(x0, x1);
            edges[y1].unset_range(x0, x1);
        }

        pub fn iter_points(&self) -> impl Iterator<Item = Vec2> {
            BoardPointIterator::new(self.n, self.points[0].clone())
        }

        fn get_rot4(&self, v1: Vec2, v2: Vec2) -> (usize, usize, u32, u32) {
            let dir = (v2 - v1).unit().to_dir() & 3;
            let v1_rot = v1.rot(dir, self.n);
            let v2_rot = v2.rot(dir, self.n);

            debug_assert!(v1_rot.y == v2_rot.y);
            let y = v1_rot.y as usize;
            let x1 = v1_rot.x as u32;
            let x2 = v2_rot.x as u32;

            let (x1, x2) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };

            (dir, y, x1, x2)
        }
    }

    pub struct BoardPointIterator {
        x: u32,
        y: usize,
        n: usize,
        points: Vec<Bitset>,
    }

    impl BoardPointIterator {
        fn new(n: usize, points: Vec<Bitset>) -> Self {
            Self {
                x: 0,
                y: 0,
                n,
                points,
            }
        }
    }

    impl Iterator for BoardPointIterator {
        type Item = Vec2;

        fn next(&mut self) -> Option<Self::Item> {
            while self.y < self.n {
                let v = self.points[self.y].v >> self.x;

                if v > 0 {
                    let d = v.trailing_zeros();
                    let x = self.x + d;
                    self.x += d + 1;
                    return Some(Vec2::new(x as i32, self.y as i32));
                }

                self.x = 0;
                self.y += 1;
            }

            None
        }
    }

    #[cfg(test)]
    mod test {
        use super::Bitset;

        #[test]
        fn set() {
            let mut b = Bitset::new(1);
            b.set(1);
            assert_eq!(b.v, 3);
        }

        #[test]
        fn unset() {
            let mut b = Bitset::new(3);
            b.unset(1);
            assert_eq!(b.v, 1);
        }

        #[test]
        fn find_next() {
            find_next_inner(1, 1, None);
            find_next_inner(1, 0, Some(0));
            find_next_inner(12, 1, Some(2));
        }

        fn find_next_inner(v: u64, begin: u32, expected: Option<u32>) {
            let actual = Bitset::new(v).find_next(begin);
            assert_eq!(actual, expected);
        }

        #[test]
        fn contains_range() {
            contains_range_inner(7, 0, 0, false);
            contains_range_inner(7, 0, 1, true);
            contains_range_inner(5, 1, 2, false);
            contains_range_inner(5, 1, 3, true);
        }

        fn contains_range_inner(v: u64, begin: u32, end: u32, expected: bool) {
            let actual = Bitset::new(v).contains_range(begin, end);
            assert_eq!(actual, expected);
        }

        #[test]
        fn set_range() {
            let mut b = Bitset::new(1);
            b.set_range(2, 4);
            assert_eq!(b.v, 13);
        }

        #[test]
        fn unset_range() {
            let mut b = Bitset::new(13);
            b.unset_range(2, 4);
            assert_eq!(b.v, 1);
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Input {
    n: usize,
    m: usize,
    p: Vec<Vec2>,
    score_coef: f64,
    since: Instant,
}

impl Input {
    fn read() -> Self {
        input! {
            n: usize,
            m: usize,
        }

        let mut p = vec![];

        for _ in 0..m {
            input! {
                x: i32,
                y: i32
            }
            p.push(Vec2::new(x, y));
        }

        let score_coef = 1e6 * (n * n) as f64 / m as f64;
        let since = Instant::now();

        let mut input = Input {
            n,
            m,
            p,
            score_coef,
            since,
        };

        let mut total_weight = 0;

        for x in 0..n {
            for y in 0..n {
                let v = Vec2::new(x as i32, y as i32);
                total_weight += input.get_weight(v);
            }
        }

        input.score_coef /= total_weight as f64;

        input
    }

    fn get_weight(&self, v: Vec2) -> i32 {
        let c = ((self.n - 1) / 2) as i32;
        let dx = v.x - c;
        let dy = v.y - c;
        dx * dx + dy * dy + 1
    }
}

#[derive(Debug, Clone)]
struct State {
    board: Board,
    rectangles: Vec<[Vec2; 4]>,
    score: i32,
}

impl State {
    fn init(input: &Input) -> Self {
        let board = Board::init(input.n, &input.p);
        let rectangles = vec![];

        let score = input.p.iter().map(|p| input.get_weight(*p)).sum();

        Self {
            board,
            rectangles,
            score,
        }
    }

    fn can_apply(&self, rectangle: &[Vec2; 4]) -> bool {
        for (i, v) in rectangle.iter().enumerate() {
            if self.board.is_occupied(*v) ^ (i != 0) {
                return false;
            }
        }

        for (i, &from) in rectangle.iter().enumerate() {
            let to = rectangle[(i + 1) % 4];
            if !self.board.can_connect(from, to) {
                return false;
            }
        }

        true
    }

    fn apply(&mut self, input: &Input, rectangle: &[Vec2; 4]) {
        self.board.add_point(rectangle[0]);
        self.rectangles.push(rectangle.clone());
        self.score += input.get_weight(rectangle[0]);

        let mut begin = 0;
        let mut edges = [Vec2::default(); 4];

        for (i, edge) in edges.iter_mut().enumerate() {
            *edge = rectangle[(i + 1) & 3] - rectangle[i];
        }

        for i in 0..4 {
            let p = &edges[i];
            if p.x > 0 && p.y >= 0 {
                begin = i;
                break;
            }
        }

        let p0 = rectangle[begin];
        let p1 = rectangle[(begin + 1) & 3];
        let p3 = rectangle[(begin + 3) & 3];

        let width = p1.x - p0.x;
        let height = p3.y - p0.y;
        let dir = if p1.y - p0.y == 0 { 0 } else { 1 };
        let height_mul = if dir == 0 { 1 } else { 2 };

        self.board
            .connect_parallel(p0, width, height * height_mul, dir);

        let (width, height) = (height, width);
        let dir = rot_cc(dir);
        self.board
            .connect_parallel(p1, width, height * height_mul, dir);
    }

    fn remove(&mut self, input: &Input, rectangle: &[Vec2; 4]) {
        self.board.remove_point(rectangle[0]);

        // rectanglesのupdateはしないことに注意！
        // self.rectangles.push(rectangle.clone());
        self.score -= input.get_weight(rectangle[0]);

        let mut begin = 0;
        let mut edges = [Vec2::default(); 4];

        for (i, edge) in edges.iter_mut().enumerate() {
            *edge = rectangle[(i + 1) & 3] - rectangle[i];
        }

        for i in 0..4 {
            let p = &edges[i];
            if p.x > 0 && p.y >= 0 {
                begin = i;
                break;
            }
        }

        let p0 = rectangle[begin];
        let p1 = rectangle[(begin + 1) & 3];
        let p3 = rectangle[(begin + 3) & 3];

        let width = p1.x - p0.x;
        let height = p3.y - p0.y;
        let dir = if p1.y - p0.y == 0 { 0 } else { 1 };
        let height_mul = if dir == 0 { 1 } else { 2 };

        self.board
            .disconnect_parallel(p0, width, height * height_mul, dir);

        let (width, height) = (height, width);
        let dir = rot_cc(dir);
        self.board
            .disconnect_parallel(p1, width, height * height_mul, dir);
    }

    fn calc_normalized_score(&self, input: &Input) -> i32 {
        (self.score as f64 * input.score_coef).round() as i32
    }

    fn calc_annealing_score(&self, input: &Input) -> f64 {
        (self.score as f64 * input.score_coef).sqrt()
    }

    fn to_output(&self) -> Output {
        Output::new(self.rectangles.clone())
    }
}

#[derive(Debug, Clone)]
struct Output {
    rectangles: Vec<[Vec2; 4]>,
}

impl Output {
    fn new(rectangles: Vec<[Vec2; 4]>) -> Self {
        Self { rectangles }
    }
}

impl std::fmt::Display for Output {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.rectangles.len())?;

        for rect in self.rectangles.iter() {
            writeln!(f)?;
            write!(f, "{} {}", rect[0].x, rect[0].y)?;

            for p in rect[1..].iter() {
                write!(f, " {} {}", p.x, p.y)?;
            }
        }

        Ok(())
    }
}

struct Parameter {
    temp_high: f64,
    temp_low: f64,
    duration: f64,
}

impl Parameter {
    fn new(input: &Input) -> Self {
        let args = std::env::args().collect::<Vec<_>>();

        let (temp_high, temp_low) = if args.len() == 3 {
            eprintln!("reading parameters from args...");
            (args[1].parse().unwrap(), args[2].parse().unwrap())
        } else {
            Self::get_best_temp(&input)
        };

        let duration_mul =
            std::env::var("DURATION_MUL").map_or_else(|_| 1.0, |val| val.parse::<f64>().unwrap());
        let duration = 4.98 * duration_mul;
        Self {
            temp_high,
            temp_low,
            duration,
        }
    }

    fn get_best_temp(input: &Input) -> (f64, f64) {
        let model = neural_network::generate_model();
        let mut best_temp0 = 1.0;
        let mut best_temp1 = 1.0;
        let mut best_score = 0.0;

        const GRID_DIV: usize = 10;

        for i in 0..=GRID_DIV {
            for j in 0..=GRID_DIV {
                let temp0 = 5.0 * 10.0f64.powf(i as f64 / GRID_DIV as f64);
                let temp1 = 10.0f64.powf(j as f64 / GRID_DIV as f64);
                let input = Self::normalize_input(input, temp0, temp1);
                let predicted_score = model.predict(&input)[0];

                if chmax!(best_score, predicted_score) {
                    best_temp0 = temp0;
                    best_temp1 = temp1;
                }
            }
        }

        eprintln!("temp: {} {}", best_temp0, best_temp1);

        (best_temp0, best_temp1)
    }

    fn normalize_input(input: &Input, temp0: f64, temp1: f64) -> Vec<f64> {
        let n = (input.n - 31) as f64 / 30.0;
        let density = input.m as f64 / (input.n * input.n) as f64 * 10.0;
        let temp0 = temp0.log10();
        let temp1 = temp1.log10();
        vec![n, density, temp0, temp1]
    }
}

fn main() {
    let input = Input::read();
    let params = Parameter::new(&input);
    eprintln!("Elapsed: {}ms", (Instant::now() - input.since).as_millis());

    let output = annealing(&input, State::init(&input), params.duration, &params).to_output();
    eprintln!("Elapsed: {}ms", (Instant::now() - input.since).as_millis());
    println!("{}", output);
}

fn annealing(input: &Input, initial_solution: State, duration: f64, params: &Parameter) -> State {
    let mut solution = initial_solution;
    let mut best_solution = solution.clone();
    let mut best_score = solution.calc_normalized_score(input);

    let mut all_iter = 0;
    let mut valid_iter = 0;
    let mut accepted_count = 0;
    let mut update_count = 0;
    let mut rng = rand_pcg::Pcg64Mcg::new(42);

    let duration_inv = 1.0 / duration;
    let since = std::time::Instant::now();

    let temp0 = params.temp_high;
    let temp1 = params.temp_low;

    const MOVIE_FRAME_COUNT: usize = 300;
    let export_movie = std::env::var("MOVIE").is_ok();
    let mut movie = vec![];

    const NOT_IMPROVED_THRESHOLD: f64 = 0.1;
    let mut last_improved = 0.0;

    let mut ls_sampler = LargeSmallSampler::new(rng.gen());

    loop {
        all_iter += 1;

        let time = (std::time::Instant::now() - since).as_secs_f64() * duration_inv;

        if time >= 1.0 {
            break;
        }

        let temp = f64::powf(temp0, 1.0 - time) * f64::powf(temp1, time);

        // 変形
        let will_removed = skip_none!(try_break_rectangles(input, &solution, &mut rng));

        if solution.rectangles.len() != 0 && will_removed.iter().all(|b| !b) {
            continue;
        }

        let state = random_greedy(input, &will_removed, &solution, &mut ls_sampler);

        // スコア計算
        let score_diff = state.calc_annealing_score(input) - solution.calc_annealing_score(input);

        if score_diff >= 0.0 || rng.gen_bool(f64::exp(score_diff as f64 / temp)) {
            // 解の更新
            accepted_count += 1;
            solution = state;

            if chmax!(best_score, solution.calc_normalized_score(input)) {
                best_solution = solution.clone();
                update_count += 1;
                last_improved = time;
            } else {
                if time - last_improved >= NOT_IMPROVED_THRESHOLD {
                    solution = best_solution.clone();
                    last_improved = time;
                }
            }
        }

        if export_movie && valid_iter % 10 == 0 {
            movie.push(solution.to_output());
        }

        valid_iter += 1;
    }

    eprintln!("===== annealing =====");
    eprintln!("score      : {}", best_score);
    eprintln!("all iter   : {}", all_iter);
    eprintln!("valid iter : {}", valid_iter);
    eprintln!("accepted   : {}", accepted_count);
    eprintln!("updated    : {}", update_count);
    eprintln!("");

    if export_movie {
        if movie.len() <= MOVIE_FRAME_COUNT {
            for output in movie {
                println!("{}", output);
            }
        } else {
            for i in 0..MOVIE_FRAME_COUNT {
                println!("{}", movie[i * movie.len() / MOVIE_FRAME_COUNT]);
            }
        }
    }

    best_solution
}

fn try_break_rectangles(
    input: &Input,
    solution: &State,
    rng: &mut rand_pcg::Pcg64Mcg,
) -> Option<Vec<bool>> {
    let size = rng.gen_range(1, input.n / 2);
    let x0 = rng.gen_range(0, input.n - size + 1);
    let y0 = rng.gen_range(0, input.n - size + 1);
    let x1 = x0 + size;
    let y1 = y0 + size;
    let count = solution.board.get_range_popcnt(x0, y0, x1, y1);

    if (solution.rectangles.len() != 0 && count == 0) || count >= 50 {
        return None;
    }

    let will_removed = solution
        .rectangles
        .iter()
        .map(|rect| {
            let p = rect[0];
            let x = p.x as usize;
            let y = p.y as usize;
            x0 <= x && x < x1 && y0 <= y && y < y1
        })
        .collect();

    Some(will_removed)
}

fn random_greedy(
    input: &Input,
    will_removed: &[bool],
    state: &State,
    sampler: &mut impl Sampler<[Vec2; 4]>,
) -> State {
    // 削除予定の矩形・それに依存する矩形を削除
    sampler.clear();
    let mut state = state.clone();
    let mut old_rectangles = Vec::with_capacity(state.rectangles.len() * 6 / 5);
    swap(&mut state.rectangles, &mut old_rectangles);

    for (&remove, rectangle) in will_removed.iter().zip(old_rectangles.iter()) {
        let remove = remove || rectangle[1..].iter().any(|v| !state.board.is_occupied(*v));
        if remove {
            state.remove(input, rectangle);
        } else {
            state.rectangles.push(*rectangle);
        }
    }

    let mut next_p = [None; DIR_COUNT];

    for p2 in state.board.iter_points() {
        for (dir, next) in next_p.iter_mut().enumerate() {
            *next = state.board.find_next(p2, dir);
        }

        for dir in 0..8 {
            let p1 = skip_none!(next_p[dir]);
            let p3 = skip_none!(next_p[rot_c(dir)]);
            let p0 = p1 + (p3 - p2);

            try_add_candidate(input, &state, p0, p1, p2, p3, sampler)
        }
    }

    // 複数回再構築をトライする
    const TRIAL_COUNT: usize = 2;
    let init_len = state.rectangles.len();
    let mut used = vec![];
    let mut best_rect = vec![];
    let mut best_score = state.calc_normalized_score(input);

    for _ in 0..TRIAL_COUNT {
        loop {
            let rectangle = if let Some(rect) = sampler.sample() {
                rect
            } else {
                break;
            };

            used.push(rectangle);

            if !state.can_apply(&rectangle) {
                continue;
            }

            state.apply(input, &rectangle);

            for (p0, p1, p2, p3) in NextPointIterator::new(&state, rectangle) {
                try_add_candidate(input, &state, p0, p1, p2, p3, sampler)
            }
        }

        if chmax!(best_score, state.calc_normalized_score(input)) {
            best_rect.clear();
            for &rect in state.rectangles[init_len..].iter() {
                best_rect.push(rect);
            }
        }

        let count = state.rectangles.len() - init_len;

        // ロールバックする
        // 初期状態から到達できないゴミが残ってしまうが、state.can_apply()で弾かれる
        // 前回選ばれた頂点は再度選ばれやすくなってしまうが、許容
        // TODO: 2回目でかつ更新したばかりの場合はロールバック不要
        for _ in 0..count {
            let rect = state.rectangles.pop().unwrap();
            state.remove(input, &rect);
        }

        while let Some(rect) = used.pop() {
            sampler.push(rect);
        }
    }

    for rect in best_rect.iter() {
        state.apply(input, rect);
    }

    state
}

fn try_add_candidate(
    input: &Input,
    state: &State,
    p0: Vec2,
    p1: Vec2,
    p2: Vec2,
    p3: Vec2,
    sampler: &mut impl Sampler<[Vec2; 4]>,
) {
    if !p0.in_map(input.n)
        || state.board.is_occupied(p0)
        || !state.board.can_connect(p1, p0)
        || !state.board.can_connect(p3, p0)
    {
        return;
    }

    let rectangle = [p0, p1, p2, p3];
    sampler.push(rectangle);
}

struct NextPointIterator<'a> {
    rectangle: [Vec2; 4],
    next: [Option<Vec2>; DIR_COUNT],
    dir: usize,
    phase: usize,
    state: &'a State,
}

impl<'a> NextPointIterator<'a> {
    fn new(state: &'a State, rectangle: [Vec2; 4]) -> Self {
        let mut next = [None; DIR_COUNT];

        for (dir, next) in next.iter_mut().enumerate() {
            *next = state.board.find_next(rectangle[0], dir);
        }

        let dir = 0;
        let phase = 0;

        Self {
            rectangle,
            next,
            dir,
            phase,
            state,
        }
    }
}

impl<'a> Iterator for NextPointIterator<'a> {
    type Item = (Vec2, Vec2, Vec2, Vec2);

    fn next(&mut self) -> Option<Self::Item> {
        if self.phase == 0 {
            let p1 = self.rectangle[0];

            while self.dir < DIR_COUNT {
                let dir = self.dir;
                self.dir += 1;

                let p2 = skip_none!(self.next[dir]);
                let p3 = skip_none!(self.state.board.find_next(p2, rot_cc(dir)));
                let p0 = p1 + (p3 - p2);
                return Some((p0, p1, p2, p3));
            }

            self.phase += 1;
            self.dir = 0;
        }

        if self.phase == 1 {
            let p2 = self.rectangle[0];

            while self.dir < DIR_COUNT {
                let dir = self.dir;
                self.dir += 1;

                let p1 = skip_none!(self.next[dir]);
                let p3 = skip_none!(self.next[rot_c(dir)]);
                let p0 = p1 + (p3 - p2);
                return Some((p0, p1, p2, p3));
            }

            self.phase += 1;
            self.dir = 0;
        }

        if self.phase == 2 {
            let p3 = self.rectangle[0];

            while self.dir < DIR_COUNT {
                let dir = self.dir;
                self.dir += 1;

                let p2 = skip_none!(self.next[dir]);
                let p1 = skip_none!(self.state.board.find_next(p2, rot_c(dir)));
                let p0 = p1 + (p3 - p2);
                return Some((p0, p1, p2, p3));
            }
        }

        None
    }
}

trait Sampler<T> {
    fn push(&mut self, item: T);
    fn sample(&mut self) -> Option<T>;
    fn clear(&mut self);
}

struct LargeSmallSampler {
    items_small: Vec<[Vec2; 4]>,
    items_large: Vec<[Vec2; 4]>,
    init: bool,
    rng: Pcg64Mcg,
}

impl LargeSmallSampler {
    fn new(seed: u128) -> Self {
        let items_small = Vec::with_capacity(32);
        let items_large = Vec::with_capacity(32);
        let init = true;
        let rng = Pcg64Mcg::new(seed);
        Self {
            items_small,
            items_large,
            init,
            rng,
        }
    }
}

impl Sampler<[Vec2; 4]> for LargeSmallSampler {
    fn push(&mut self, item: [Vec2; 4]) {
        let p0 = item[0];
        let p1 = item[1];
        let p3 = item[3];

        let v0 = p1 - p0;
        let v1 = p3 - p0;
        let norm0 = v0.norm2_sq();
        let norm1 = v1.norm2_sq();

        if (norm0 == 1 && norm1 == 1) || (norm0 == 2 && norm1 == 2) {
            self.items_small.push(item);
        } else {
            self.items_large.push(item);
        }
    }

    fn sample(&mut self) -> Option<[Vec2; 4]> {
        let len_small = self.items_small.len();
        let len_large = self.items_large.len();

        if self.init {
            self.init = false;

            if len_small + len_large == 0 {
                return None;
            }

            let i = self.rng.gen_range(0, len_small + len_large);

            if i < len_small {
                return Some(self.items_small.swap_remove(i));
            } else {
                return Some(self.items_large.swap_remove(i - len_small));
            }
        }

        if len_small > 0 {
            let i = self.rng.gen_range(0, len_small);
            Some(self.items_small.swap_remove(i))
        } else if len_large > 0 {
            let i = self.rng.gen_range(0, len_large);
            Some(self.items_large.swap_remove(i))
        } else {
            None
        }
    }

    fn clear(&mut self) {
        self.items_small.clear();
        self.items_large.clear();
        self.init = true;
    }
}

#[allow(dead_code)]
mod vector {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
    pub struct Vec2 {
        pub x: i32,
        pub y: i32,
    }

    impl Vec2 {
        pub const fn new(x: i32, y: i32) -> Self {
            Self { x, y }
        }

        pub fn in_map(&self, n: usize) -> bool {
            let n = n as u32;
            (self.x as u32) < n && (self.y as u32) < n
        }

        pub fn rot(&self, dir: usize, n: usize) -> Self {
            let mut x = self.x;
            let mut y = self.y;
            let n = n as i32;

            // 180°回転
            if ((dir >> 2) & 1) > 0 {
                x = n - 1 - x;
                y = n - 1 - y;
            }

            // 90°回転
            if ((dir >> 1) & 1) > 0 {
                let xx = y;
                let yy = n - 1 - x;
                x = xx;
                y = yy;
            }

            // 45°回転
            if (dir & 1) > 0 {
                let xx = (x + y) >> 1;
                let yy = n - 1 - x + y;
                x = xx;
                y = yy;
            }

            Vec2::new(x, y)
        }

        pub fn cross(&self, rhs: Vec2) -> i32 {
            self.x * rhs.y - self.y * rhs.x
        }

        pub fn unit(&self) -> Self {
            debug_assert!(((self.x == 0) ^ (self.y == 0)) || self.x.abs() == self.y.abs());
            Self::new(self.x.signum(), self.y.signum())
        }

        pub fn norm2(&self) -> f64 {
            let sq = self.norm2_sq() as f64;
            sq.sqrt()
        }

        pub fn norm2_sq(&self) -> i32 {
            self.x * self.x + self.y * self.y
        }

        pub fn to_dir(&self) -> usize {
            let abs = self.x.abs() + self.y.abs();
            debug_assert!(1 <= abs && abs <= 2);
            const DIRECTIONS: [usize; 9] = [5, 6, 7, 4, !0, 0, 3, 2, 1];
            DIRECTIONS[((self.y + 1) * 3 + (self.x + 1)) as usize]
        }
    }

    impl std::ops::Add for Vec2 {
        type Output = Vec2;

        fn add(self, rhs: Self) -> Self::Output {
            Vec2::new(self.x + rhs.x, self.y + rhs.y)
        }
    }

    impl std::ops::AddAssign for Vec2 {
        fn add_assign(&mut self, rhs: Self) {
            self.x += rhs.x;
            self.y += rhs.y;
        }
    }

    impl std::ops::Sub for Vec2 {
        type Output = Vec2;

        fn sub(self, rhs: Self) -> Self::Output {
            Vec2::new(self.x - rhs.x, self.y - rhs.y)
        }
    }

    impl std::ops::SubAssign for Vec2 {
        fn sub_assign(&mut self, rhs: Self) {
            self.x -= rhs.x;
            self.y -= rhs.y;
        }
    }

    impl std::ops::Mul<i32> for Vec2 {
        type Output = Vec2;

        fn mul(self, rhs: i32) -> Self::Output {
            let x = self.x * rhs;
            let y = self.y * rhs;
            Self::new(x, y)
        }
    }

    impl std::ops::MulAssign<i32> for Vec2 {
        fn mul_assign(&mut self, rhs: i32) {
            self.x *= rhs;
            self.y *= rhs;
        }
    }

    impl std::ops::Neg for Vec2 {
        type Output = Vec2;

        fn neg(self) -> Self::Output {
            Vec2::new(-self.x, -self.y)
        }
    }

    impl std::fmt::Display for Vec2 {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "({}, {})", self.x, self.y)
        }
    }

    pub const UNITS: [Vec2; 8] = [
        Vec2::new(1, 0),
        Vec2::new(1, 1),
        Vec2::new(0, 1),
        Vec2::new(-1, 1),
        Vec2::new(-1, 0),
        Vec2::new(-1, -1),
        Vec2::new(0, -1),
        Vec2::new(1, -1),
    ];

    pub const DIR_COUNT: usize = 8;

    pub const fn inv(dir: usize) -> usize {
        dir ^ 4
    }

    pub const fn rot_cc(dir: usize) -> usize {
        (dir + 2) % 8
    }

    pub const fn rot_c(dir: usize) -> usize {
        (dir + 6) % 8
    }

    #[cfg(test)]
    mod test {
        use super::Vec2;

        #[test]
        fn rot180() {
            let v = Vec2::new(2, 1);
            let v = v.rot(4, 4);
            assert_eq!(v, Vec2::new(1, 2));
        }

        #[test]
        fn rot90() {
            let v = Vec2::new(2, 1);
            let v = v.rot(2, 4);
            assert_eq!(v, Vec2::new(1, 1));
        }

        #[test]
        fn rot45() {
            let v = Vec2::new(2, 1);
            let v = v.rot(1, 4);
            assert_eq!(v, Vec2::new(1, 2));

            let v = Vec2::new(2, 0);
            let v = v.rot(1, 4);
            assert_eq!(v, Vec2::new(1, 1));
        }
    }
}

mod base64 {
    pub(super) fn to_f64(data: &[u8]) -> Vec<f64> {
        const BASE64_MAP: &[u8] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut stream = vec![];

        let mut cursor = 0;

        while cursor + 4 <= data.len() {
            let mut buffer = 0u32;

            for i in 0..4 {
                let c = data[cursor + i];
                let shift = 6 * (3 - i);

                for (i, &d) in BASE64_MAP.iter().enumerate() {
                    if c == d {
                        buffer |= (i as u32) << shift;
                    }
                }
            }

            for i in 0..3 {
                let shift = 8 * (2 - i);
                let value = (buffer >> shift) as u8;
                stream.push(value);
            }

            cursor += 4;
        }

        let mut result = vec![];
        cursor = 0;

        while cursor + 8 <= stream.len() {
            let p = stream.as_ptr() as *const f64;
            let x = unsafe { *p.offset(cursor as isize / 8) };
            result.push(x);
            cursor += 8;
        }

        result
    }
}

mod neural_network {
    use crate::base64;

    pub(super) struct NeuralNetwork {
        in_weight: Vec<Vec<f64>>,
        in_bias: Vec<f64>,
        hidden1_weight: Vec<Vec<f64>>,
        hidden1_bias: Vec<f64>,
        hidden2_weight: Vec<Vec<f64>>,
        hidden2_bias: Vec<f64>,
        hidden3_weight: Vec<Vec<f64>>,
        hidden3_bias: Vec<f64>,
        out_weight: Vec<Vec<f64>>,
        out_bias: Vec<f64>,
    }

    impl NeuralNetwork {
        pub(super) fn predict(&self, x: &[f64]) -> Vec<f64> {
            let x = multiply_matrix(&self.in_weight, x);
            let x = add_vector(&self.in_bias, &x);
            let x = apply(&x, relu);

            let x = multiply_matrix(&self.hidden1_weight, &x);
            let x = add_vector(&self.hidden1_bias, &x);
            let x = apply(&x, relu);

            let x = multiply_matrix(&self.hidden2_weight, &x);
            let x = add_vector(&self.hidden2_bias, &x);
            let x = apply(&x, relu);

            let x = multiply_matrix(&self.hidden3_weight, &x);
            let x = add_vector(&self.hidden3_bias, &x);
            let x = apply(&x, relu);

            let x = multiply_matrix(&self.out_weight, &x);
            let x = add_vector(&self.out_bias, &x);

            x
        }
    }

    pub(super) fn generate_model() -> NeuralNetwork {
        const WEIGHT_BASE64: &[u8] = b"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";
        const INPUT_SIZE: usize = 4;
        const HIDDEN_SIZE: usize = 16;
        const OUTPUT_SIZE: usize = 1;
        let weight = base64::to_f64(WEIGHT_BASE64);
        let mut cursor = 0;

        let in_weight = read_matrix(HIDDEN_SIZE, INPUT_SIZE, &weight, &mut cursor);
        let in_bias = read_vector(HIDDEN_SIZE, &weight, &mut cursor);
        let hidden1_weight = read_matrix(HIDDEN_SIZE, HIDDEN_SIZE, &weight, &mut cursor);
        let hidden1_bias = read_vector(HIDDEN_SIZE, &weight, &mut cursor);
        let hidden2_weight = read_matrix(HIDDEN_SIZE, HIDDEN_SIZE, &weight, &mut cursor);
        let hidden2_bias = read_vector(HIDDEN_SIZE, &weight, &mut cursor);
        let hidden3_weight = read_matrix(HIDDEN_SIZE, HIDDEN_SIZE, &weight, &mut cursor);
        let hidden3_bias = read_vector(HIDDEN_SIZE, &weight, &mut cursor);
        let out_weight = read_matrix(OUTPUT_SIZE, HIDDEN_SIZE, &weight, &mut cursor);
        let out_bias = read_vector(OUTPUT_SIZE, &weight, &mut cursor);

        NeuralNetwork {
            in_weight,
            in_bias,
            hidden1_weight,
            hidden1_bias,
            hidden2_weight,
            hidden2_bias,
            hidden3_weight,
            hidden3_bias,
            out_weight,
            out_bias,
        }
    }

    fn read_matrix(row: usize, col: usize, data: &[f64], cursor: &mut usize) -> Vec<Vec<f64>> {
        let mut matrix = Vec::with_capacity(row);

        for _ in 0..row {
            let vector = read_vector(col, data, cursor);
            matrix.push(vector);
        }

        matrix
    }

    fn read_vector(n: usize, data: &[f64], cursor: &mut usize) -> Vec<f64> {
        let mut vector = Vec::with_capacity(n);

        for _ in 0..n {
            vector.push(data[*cursor]);
            *cursor += 1;
        }

        vector
    }

    fn multiply_matrix(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(matrix.len());

        for line in matrix.iter() {
            debug_assert!(line.len() == vector.len());
            let mut sum = 0.0;

            for (x, y) in line.iter().zip(vector.iter()) {
                sum += x * y;
            }

            result.push(sum);
        }

        result
    }

    fn add_vector(vector1: &[f64], vector2: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(vector1.len());
        debug_assert!(vector1.len() == vector2.len());

        for (x, y) in vector1.iter().zip(vector2.iter()) {
            result.push(x + y);
        }

        result
    }

    fn apply<F>(vector: &[f64], f: F) -> Vec<f64>
    where
        F: Fn(f64) -> f64,
    {
        let mut result = Vec::with_capacity(vector.len());

        for x in vector.iter() {
            result.push(f(*x));
        }

        result
    }

    fn relu(x: f64) -> f64 {
        x.max(0.0)
    }
}
