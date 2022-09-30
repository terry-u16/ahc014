use std::{
    mem::{swap, MaybeUninit},
    time::Instant,
};

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
                let p = self.points.get_unchecked(dir);
                let v_rot = v.rot(dir, self.n);
                let next = p
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
            unsafe {
                self.points
                    .get_unchecked(0)
                    .get_unchecked(v1.y as usize)
                    .at(v1.x as u32)
            }
        }

        pub fn can_connect(&self, v1: Vec2, v2: Vec2) -> bool {
            unsafe {
                let (dir, y, x1, x2) = self.get_rot4(v1, v2);
                let has_point = self
                    .points
                    .get_unchecked(dir)
                    .get_unchecked(y)
                    .contains_range(x1 + 1, x2);
                let has_edge = self
                    .edges
                    .get_unchecked(dir)
                    .get_unchecked(y)
                    .contains_range(x1, x2);
                !has_point && !has_edge
            }
        }

        pub fn add_point(&mut self, v: Vec2) {
            for dir in 0..DIR_COUNT {
                let v = v.rot(dir, self.n);
                unsafe {
                    self.points
                        .get_unchecked_mut(dir)
                        .get_unchecked_mut(v.y as usize)
                        .set(v.x as u32);
                }
            }
        }

        pub fn remove_point(&mut self, v: Vec2) {
            for dir in 0..DIR_COUNT {
                let v = v.rot(dir, self.n);
                unsafe {
                    self.points
                        .get_unchecked_mut(dir)
                        .get_unchecked_mut(v.y as usize)
                        .unset(v.x as u32);
                }
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
            unsafe {
                let edges = self.edges.get_unchecked_mut(dir);
                edges.get_unchecked_mut(y0).set_range(x0, x1);
                edges.get_unchecked_mut(y1).set_range(x0, x1);
            }
        }

        pub fn disconnect_parallel(&mut self, v0: Vec2, width: i32, height: i32, dir: usize) {
            let v0 = v0.rot(dir, self.n);
            let y0 = v0.y as usize;
            let x0 = v0.x as u32;
            let y1 = (y0 as i32 + height) as usize;
            let x1 = x0 + width as u32;
            unsafe {
                let edges = self.edges.get_unchecked_mut(dir);
                edges.get_unchecked_mut(y0).unset_range(x0, x1);
                edges.get_unchecked_mut(y1).unset_range(x0, x1);
            }
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
            unsafe {
                let to = *rectangle.get_unchecked((i + 1) % 4);
                if !self.board.can_connect(from, to) {
                    return false;
                }
            }
        }

        true
    }

    fn apply(&mut self, input: &Input, rectangle: &[Vec2; 4]) {
        unsafe {
            let p = *rectangle.get_unchecked(0);
            self.board.add_point(p);
            self.rectangles.push(rectangle.clone());
            self.score += input.get_weight(p);

            let mut begin = 0;
            let mut edges: [MaybeUninit<Vec2>; 4] = MaybeUninit::uninit().assume_init();

            for (i, edge) in edges.iter_mut().enumerate() {
                *edge = MaybeUninit::new(
                    *rectangle.get_unchecked((i + 1) & 3) - *rectangle.get_unchecked(i),
                );
            }

            let edges: [Vec2; 4] = std::mem::transmute(edges);

            for i in 0..4 {
                let p = edges.get_unchecked(i);
                if p.x > 0 && p.y >= 0 {
                    begin = i;
                    break;
                }
            }

            let p0 = *rectangle.get_unchecked(begin);
            let p1 = *rectangle.get_unchecked((begin + 1) & 3);
            let p3 = *rectangle.get_unchecked((begin + 3) & 3);

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
    }

    fn remove(&mut self, input: &Input, rectangle: &[Vec2; 4]) {
        unsafe {
            let p = *rectangle.get_unchecked(0);
            self.board.remove_point(p);

            // rectanglesのupdateはしないことに注意！
            // self.rectangles.push(rectangle.clone());
            self.score -= input.get_weight(p);

            let mut begin = 0;
            let mut edges: [MaybeUninit<Vec2>; 4] = MaybeUninit::uninit().assume_init();

            for (i, edge) in edges.iter_mut().enumerate() {
                *edge = MaybeUninit::new(
                    *rectangle.get_unchecked((i + 1) & 3) - *rectangle.get_unchecked(i),
                );
            }

            let edges: [Vec2; 4] = std::mem::transmute(edges);

            for i in 0..4 {
                let p = edges.get_unchecked(i);
                if p.x > 0 && p.y >= 0 {
                    begin = i;
                    break;
                }
            }

            let p0 = *rectangle.get_unchecked(begin);
            let p1 = *rectangle.get_unchecked((begin + 1) & 3);
            let p3 = *rectangle.get_unchecked((begin + 3) & 3);

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

        const GRID_DIV: usize = 30;

        for i in 0..=GRID_DIV {
            for j in 0..=GRID_DIV {
                let temp0 = 5.0 * 5.0f64.powf(i as f64 / GRID_DIV as f64);
                let temp1 = 2.0 * 5.0f64.powf(j as f64 / GRID_DIV as f64);

                if temp0 < temp1 {
                    continue;
                }

                let input = Self::normalize_input(input, temp0, temp1);
                let predicted_score = model.predict(&input)[0];

                if chmax!(best_score, predicted_score) {
                    best_temp0 = temp0;
                    best_temp1 = temp1;
                }
            }
        }

        eprintln!("temp: {} {}", best_temp0, best_temp1);
        eprintln!("predicted_score: {}", best_score * 1e6);

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

    let threshold = ((solution.rectangles.len() + input.m) / 3).min(30);
    if (solution.rectangles.len() != 0 && count == 0) || count >= threshold {
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

static mut USED_IN_GREEDY: Vec<[Vec2; 4]> = Vec::new();
static mut BEST_RECT_IN_GREEDY: Vec<[Vec2; 4]> = Vec::new();

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
    const TRIAL_COUNT: usize = 3;
    let init_len = state.rectangles.len();
    let mut best_score = state.calc_normalized_score(input);
    let mut no_apply = false;

    unsafe {
        BEST_RECT_IN_GREEDY.clear();
        USED_IN_GREEDY.clear();

        for trial in 0..TRIAL_COUNT {
            sampler.init();

            loop {
                let rectangle = if let Some(rect) = sampler.sample() {
                    rect
                } else {
                    break;
                };

                USED_IN_GREEDY.push(rectangle);

                if !state.can_apply(&rectangle) {
                    continue;
                }

                state.apply(input, &rectangle);

                for (p0, p1, p2, p3) in NextPointIterator::new(&state, rectangle) {
                    try_add_candidate(input, &state, p0, p1, p2, p3, sampler)
                }
            }

            if chmax!(best_score, state.calc_normalized_score(input)) {
                if trial == TRIAL_COUNT - 1 {
                    no_apply = true;
                    break;
                }

                BEST_RECT_IN_GREEDY.clear();
                for &rect in state.rectangles[init_len..].iter() {
                    BEST_RECT_IN_GREEDY.push(rect);
                }
            }

            let count = state.rectangles.len() - init_len;

            // ロールバックする
            // 初期状態から到達できないゴミが残ってしまうが、state.can_apply()で弾かれる
            // 前回選ばれた頂点は再度選ばれやすくなってしまうが、許容
            for _ in 0..count {
                let rect = state.rectangles.pop().unwrap();
                state.remove(input, &rect);
            }

            while let Some(rect) = USED_IN_GREEDY.pop() {
                sampler.push(rect);
            }
        }

        if !no_apply {
            for rect in BEST_RECT_IN_GREEDY.iter() {
                state.apply(input, rect);
            }
        }
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
    fn init(&mut self);
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
    }

    fn init(&mut self) {
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
            let x = apply(&x, sigmoid);
            let x = apply(&x, mul3);

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

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + f64::exp(-x))
    }

    fn mul3(x: f64) -> f64 {
        x * 3.0
    }
}
