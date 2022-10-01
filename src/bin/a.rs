use std::{mem::MaybeUninit, time::Instant};

use bitboard::Board;
use proconio::*;
use rand::prelude::*;

use crate::vector::{rot_cc, Vec2};
type Rectangle = [Vec2; 4];

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
    use crate::vector::{rot_cc, Vec2, DIR_COUNT, UNITS};

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

        fn flip(&mut self, i: u32) {
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

        fn flip_range(&mut self, begin: u32, end: u32) {
            self.v ^= Self::get_range_mask(begin, end);
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
        root_edges: Vec<u8>,
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
                    points[p.y as usize].flip(p.x as u32);
                }
            }

            let root_edges = vec![0; n * n];

            Self {
                n,
                points,
                edges,
                root_edges,
            }
        }

        pub fn find_next(&self, v: Vec2, dir: usize) -> Option<Vec2> {
            unsafe {
                let i = (v.y as u32 * self.n as u32 + v.x as u32) as usize;
                if (self.root_edges.get_unchecked(i) & (1 << dir)) > 0 {
                    return None;
                }

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
                        .flip(v.x as u32);
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
                        .flip(v.x as u32);
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

        pub fn flip_parallel_edges(&mut self, v0: Vec2, width: i32, height: i32, dir: usize) {
            let v0 = v0.rot(dir, self.n);
            let y0 = v0.y as usize;
            let x0 = v0.x as u32;
            let y1 = (y0 as i32 + height) as usize;
            let x1 = x0 + width as u32;
            unsafe {
                let edges = self.edges.get_unchecked_mut(dir);
                edges.get_unchecked_mut(y0).flip_range(x0, x1);
                edges.get_unchecked_mut(y1).flip_range(x0, x1);
            }
        }

        pub fn flip_root_edge2(&mut self, v: Vec2, dir: usize) {
            let i = (v.y as u32 * self.n as u32 + v.x as u32) as usize;
            unsafe {
                let e = self.root_edges.get_unchecked_mut(i);
                *e ^= 1 << dir;
                *e ^= 1 << rot_cc(dir);
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
            b.flip(1);
            assert_eq!(b.v, 3);
        }

        #[test]
        fn unset() {
            let mut b = Bitset::new(3);
            b.flip(1);
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
            b.flip_range(2, 4);
            assert_eq!(b.v, 13);
        }

        #[test]
        fn unset_range() {
            let mut b = Bitset::new(13);
            b.flip_range(2, 4);
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
    rectangles: Vec<Rectangle>,
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

    fn can_apply(&self, rectangle: &Rectangle) -> bool {
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

    fn apply(&mut self, input: &Input, rectangle: &Rectangle) {
        unsafe {
            let p = *rectangle.get_unchecked(0);
            self.board.add_point(p);
            self.rectangles.push(rectangle.clone());
            self.score += input.get_weight(p);

            self.flip_rectangle(rectangle);
        }
    }

    fn remove(&mut self, input: &Input, rectangle: &Rectangle) {
        unsafe {
            let p = *rectangle.get_unchecked(0);
            self.board.remove_point(p);

            // rectanglesのupdateはしないことに注意！
            // self.rectangles.push(rectangle.clone());
            self.score -= input.get_weight(p);

            self.flip_rectangle(rectangle);
        }
    }

    fn flip_rectangle(&mut self, rectangle: &Rectangle) {
        unsafe {
            let mut begin = 0;
            let mut edges: [MaybeUninit<Vec2>; 4] = MaybeUninit::uninit().assume_init();

            for (i, edge) in edges.iter_mut().enumerate() {
                *edge = MaybeUninit::new(
                    *rectangle.get_unchecked((i + 1) & 3) - *rectangle.get_unchecked(i),
                );
            }

            let edges: Rectangle = std::mem::transmute(edges);

            for i in 0..4 {
                let p = edges.get_unchecked(i);
                if p.x > 0 && p.y >= 0 {
                    begin = i;
                    break;
                }
            }

            let p0 = *rectangle.get_unchecked(begin);
            let p1 = *rectangle.get_unchecked((begin + 1) & 3);
            let p2 = *rectangle.get_unchecked((begin + 2) & 3);
            let p3 = *rectangle.get_unchecked((begin + 3) & 3);

            let width = p1.x - p0.x;
            let height = p3.y - p0.y;
            let dir = if p1.y - p0.y == 0 { 0 } else { 1 };
            let height_mul = if dir == 0 { 1 } else { 2 };

            self.board.flip_root_edge2(p0, dir);
            self.board.flip_root_edge2(p1, dir + 2);
            self.board.flip_root_edge2(p2, dir + 4);
            self.board.flip_root_edge2(p3, dir + 6);

            self.board
                .flip_parallel_edges(p0, width, height * height_mul, dir);
            let (width, height) = (height, width);
            let dir = rot_cc(dir);
            self.board
                .flip_parallel_edges(p1, width, height * height_mul, dir);
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
    rectangles: Vec<Rectangle>,
}

impl Output {
    fn new(rectangles: Vec<Rectangle>) -> Self {
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
    eprintln!("Elapsed: {}us", (Instant::now() - input.since).as_micros());

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

    let mut ls_sampler = greedy::LargeSmallSampler::new(rng.gen());

    loop {
        all_iter += 1;

        let time = (std::time::Instant::now() - since).as_secs_f64() * duration_inv;

        if time >= 1.0 {
            break;
        }

        let temp = f64::powf(temp0, 1.0 - time) * f64::powf(temp1, time);

        // 変形
        let will_removed = skip_none!(greedy::try_break_rectangles(input, &solution, &mut rng));
        let state = greedy::random_greedy(input, &will_removed, &solution, &mut ls_sampler);

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

mod greedy {
    use rand::prelude::*;
    use rand_pcg::Pcg64Mcg;
    use std::mem::swap;

    use crate::{
        vector::{rot_c, rot_cc, Vec2, DIR_COUNT},
        Input, Rectangle, State,
    };

    pub(crate) fn try_break_rectangles(
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

        const THRESHOLD: usize = 30;
        if (solution.rectangles.len() != 0 && count == 0) || count >= THRESHOLD {
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
            .collect::<Vec<_>>();

        if solution.rectangles.len() != 0 && will_removed.iter().all(|b| !b) {
            None
        } else {
            Some(will_removed)
        }
    }

    static mut USED_IN_GREEDY: Vec<Rectangle> = Vec::new();
    static mut BEST_RECT_IN_GREEDY: Vec<Rectangle> = Vec::new();

    pub(crate) fn random_greedy(
        input: &Input,
        will_removed: &[bool],
        state: &State,
        sampler: &mut impl Sampler<Rectangle>,
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
        sampler: &mut impl Sampler<Rectangle>,
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
        rectangle: Rectangle,
        next: [Option<Vec2>; DIR_COUNT],
        dir: usize,
        phase: usize,
        state: &'a State,
    }

    impl<'a> NextPointIterator<'a> {
        fn new(state: &'a State, rectangle: Rectangle) -> Self {
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

    pub(crate) trait Sampler<T> {
        fn push(&mut self, item: T);
        fn sample(&mut self) -> Option<T>;
        fn clear(&mut self);
        fn init(&mut self);
    }

    pub(crate) struct LargeSmallSampler {
        items_small: Vec<Rectangle>,
        items_large: Vec<Rectangle>,
        init: bool,
        rng: Pcg64Mcg,
    }

    impl LargeSmallSampler {
        pub(crate) fn new(seed: u128) -> Self {
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

    impl Sampler<Rectangle> for LargeSmallSampler {
        fn push(&mut self, item: Rectangle) {
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

        fn sample(&mut self) -> Option<Rectangle> {
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
    use itertools::Itertools;
    use nalgebra::{MatrixMN, VectorN, U1, U16, U4};

    use crate::base64;

    const INPUT_SIZE: usize = 4;
    const HIDDEN_SIZE: usize = 16;
    const OUTPUT_SIZE: usize = 1;
    type InputSizeType = U4;
    type HiddenSizeType = U16;
    type OutputSizeType = U1;

    pub(super) struct NeuralNetwork {
        in_weight: MatrixMN<f64, HiddenSizeType, InputSizeType>,
        in_bias: VectorN<f64, HiddenSizeType>,
        hidden1_weight: MatrixMN<f64, HiddenSizeType, HiddenSizeType>,
        hidden1_bias: VectorN<f64, HiddenSizeType>,
        hidden2_weight: MatrixMN<f64, HiddenSizeType, HiddenSizeType>,
        hidden2_bias: VectorN<f64, HiddenSizeType>,
        hidden3_weight: MatrixMN<f64, HiddenSizeType, HiddenSizeType>,
        hidden3_bias: VectorN<f64, HiddenSizeType>,
        out_weight: MatrixMN<f64, OutputSizeType, HiddenSizeType>,
        out_bias: VectorN<f64, OutputSizeType>,
    }

    impl NeuralNetwork {
        pub(super) fn predict(&self, x: &[f64]) -> Vec<f64> {
            let x = VectorN::<f64, InputSizeType>::from_row_slice(x);
            let mut x = &self.in_weight * x;
            x = &self.in_bias + x;
            x.apply(relu);

            x = &self.hidden1_weight * x;
            x = &self.hidden1_bias + x;
            x.apply(relu);

            x = &self.hidden2_weight * x;
            x = &self.hidden2_bias + x;
            x.apply(relu);

            x = &self.hidden3_weight * x;
            x = &self.hidden3_bias + x;
            x.apply(relu);

            let mut x = &self.out_weight * x;
            x = &self.out_bias + x;
            x.apply(sigmoid);
            x.apply(mul3);

            x.iter().copied().collect_vec()
        }
    }

    macro_rules! as_matrix {
        ($r: ty, $c: ty, $v: expr) => {
            MatrixMN::<f64, $r, $c>::from_row_slice($v)
        };
    }

    macro_rules! as_vector {
        ($d: ty, $v: expr) => {
            VectorN::<f64, $d>::from_row_slice($v)
        };
    }

    pub(super) fn generate_model() -> NeuralNetwork {
        const WEIGHT_BASE64: &[u8] = b"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";
        let weight = base64::to_f64(WEIGHT_BASE64);
        let mut cursor = 0;

        type N1T = InputSizeType;
        type N2T = HiddenSizeType;
        type N3T = OutputSizeType;
        const N1: usize = INPUT_SIZE;
        const N2: usize = HIDDEN_SIZE;
        const N3: usize = OUTPUT_SIZE;
        let in_weight = as_matrix!(N2T, N1T, &read_as_vec(N2 * N1, &weight, &mut cursor));
        let in_bias = as_vector!(N2T, &read_as_vec(N2, &weight, &mut cursor));
        let hidden1_weight = as_matrix!(N2T, N2T, &read_as_vec(N2 * N2, &weight, &mut cursor));
        let hidden1_bias = as_vector!(N2T, &read_as_vec(N2, &weight, &mut cursor));
        let hidden2_weight = as_matrix!(N2T, N2T, &read_as_vec(N2 * N2, &weight, &mut cursor));
        let hidden2_bias = as_vector!(N2T, &read_as_vec(N2, &weight, &mut cursor));
        let hidden3_weight = as_matrix!(N2T, N2T, &read_as_vec(N2 * N2, &weight, &mut cursor));
        let hidden3_bias = as_vector!(N2T, &read_as_vec(N2, &weight, &mut cursor));
        let out_weight = as_matrix!(N3T, N2T, &read_as_vec(N3 * N2, &weight, &mut cursor));
        let out_bias = as_vector!(N3T, &read_as_vec(N3, &weight, &mut cursor));

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

    fn read_as_vec(n: usize, data: &[f64], cursor: &mut usize) -> Vec<f64> {
        let mut vector = Vec::with_capacity(n);

        for _ in 0..n {
            vector.push(data[*cursor]);
            *cursor += 1;
        }

        vector
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
