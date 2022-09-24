use std::time::Instant;

use bitboard::Board;
#[allow(unused_imports)]
use proconio::*;
#[allow(unused_imports)]
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use sample::WeightedSampler;
use vector::{DIR_COUNT, UNITS};

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
        #[inline]
        #[allow(dead_code)]
        const fn new(v: u64) -> Self {
            Self { v }
        }

        #[inline]
        fn at(&self, i: u32) -> bool {
            ((self.v >> i) & 1) > 0
        }

        #[inline]
        fn set(&mut self, i: u32) {
            debug_assert!(((self.v >> i) & 1) == 0);
            self.v ^= 1 << i;
        }

        #[inline]
        fn unset(&mut self, i: u32) {
            debug_assert!(((self.v >> i) & 1) > 0);
            self.v ^= 1 << i;
        }

        #[inline]
        fn find_next(&self, begin: u32) -> Option<u32> {
            let v = self.v >> begin;
            if v == 0 {
                None
            } else {
                let tz = v.trailing_zeros();
                Some(begin + tz)
            }
        }

        #[inline]
        fn contains_range(&self, begin: u32, end: u32) -> bool {
            debug_assert!(begin <= end);
            (self.v & Self::get_range_mask(begin, end)) > 0
        }

        #[inline]
        fn set_range(&mut self, begin: u32, end: u32) {
            debug_assert!(!self.contains_range(begin, end));
            self.v ^= Self::get_range_mask(begin, end);
        }

        #[inline]
        fn unset_range(&mut self, begin: u32, end: u32) {
            let mask = Self::get_range_mask(begin, end);
            debug_assert!((self.v & mask) == mask);
            self.v ^= mask;
        }

        #[inline]
        fn get_range_popcnt(&self, begin: u32, end: u32) -> u32 {
            let mask = Self::get_range_mask(begin, end);
            (self.v & mask).count_ones()
        }

        #[inline]
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

        #[inline]
        pub fn find_next(&self, v: Vec2, dir: usize) -> Option<Vec2> {
            let v_rot = v.rot(dir, self.n);
            let next = self.points[dir][v_rot.y as usize].find_next(v_rot.x as u32 + 1);

            if let Some(next) = next {
                let unit = UNITS[dir];
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

                let has_edge = self.edges[dir][y as usize].contains_range(x1 as u32, x2 as u32);

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

        #[inline]
        pub fn is_occupied(&self, v1: Vec2) -> bool {
            self.points[0][v1.y as usize].at(v1.x as u32)
        }

        #[inline]
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

        #[inline]
        pub fn connect(&mut self, v1: Vec2, v2: Vec2) {
            self.connect_inner(v1, v2);
        }

        #[inline]
        pub fn disconnect(&mut self, v1: Vec2, v2: Vec2) {
            self.disconnect_inner(v1, v2);
        }

        pub fn get_range_popcnt(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> usize {
            let mut count = 0;

            for y in y0..y1 {
                count += self.points[0][y].get_range_popcnt(x0 as u32, x1 as u32);
            }

            count as usize
        }

        #[inline]
        fn connect_inner(&mut self, v1: Vec2, v2: Vec2) {
            let (dir, y, x1, x2) = self.get_rot4(v1, v2);
            self.edges[dir][y].set_range(x1, x2);
        }

        #[inline]
        fn disconnect_inner(&mut self, v1: Vec2, v2: Vec2) {
            let (dir, y, x1, x2) = self.get_rot4(v1, v2);
            self.edges[dir][y].unset_range(x1, x2);
        }

        #[inline]
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
    points: Vec<Vec2>,
    board: Board,
    rectangles: Vec<[Vec2; 4]>,
    score: i32,
}

#[allow(dead_code)]
impl State {
    fn init(input: &Input) -> Self {
        let points = input.p.clone();
        let board = Board::init(input.n, &input.p);
        let rectangles = vec![];

        let score = points.iter().map(|p| input.get_weight(*p)).sum();

        Self {
            points,
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
        self.points.push(rectangle[0]);
        self.board.add_point(rectangle[0]);
        self.rectangles.push(rectangle.clone());
        self.score += input.get_weight(rectangle[0]);

        for (i, &from) in rectangle.iter().enumerate() {
            let to = rectangle[(i + 1) % 4];
            self.board.connect(from, to);
        }
    }

    fn revert(&mut self, input: &Input) {
        let rectangle = self.rectangles.pop().unwrap();
        self.points.pop();
        self.board.remove_point(rectangle[0]);
        self.score -= input.get_weight(rectangle[0]);

        for (i, &from) in rectangle.iter().enumerate() {
            let to = rectangle[(i + 1) % 4];
            self.board.disconnect(from, to);
        }
    }

    fn calc_normalized_score(&self, input: &Input) -> i32 {
        (self.score as f64 * input.score_coef).round() as i32
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

fn main() {
    let input = Input::read();
    let output = annealing(&input, State::init(&input), 4.98).to_output();
    eprintln!("Elapsed: {}ms", (Instant::now() - input.since).as_millis());
    println!("{}", output);
}

fn annealing(input: &Input, initial_solution: State, duration: f64) -> State {
    let mut solution = initial_solution;
    let mut best_solution = solution.clone();
    let mut current_score = solution.calc_normalized_score(input);
    let mut best_score = current_score;

    let mut all_iter = 0;
    let mut valid_iter = 0;
    let mut accepted_count = 0;
    let mut update_count = 0;
    let mut rng = rand_pcg::Pcg64Mcg::new(42);

    let duration_inv = 1.0 / duration;
    let since = std::time::Instant::now();

    let temp0 = 3e4;
    let temp1 = 3e3;

    const MOVIE_FRAME_COUNT: usize = 300;
    let export_movie = std::env::var("MOVIE").is_ok();
    let mut movie = vec![];

    const NOT_IMPROVED_THRESHOLD: usize = 10000;
    let mut not_improved = 0;

    loop {
        all_iter += 1;

        let time = (std::time::Instant::now() - since).as_secs_f64() * duration_inv;

        if time >= 1.0 {
            break;
        }

        let temp = f64::powf(temp0, 1.0 - time) * f64::powf(temp1, time);

        // 変形
        let init_rectangles = if rng.gen_bool(0.0) {
            try_break_rectangles(input, &solution, &mut rng)
        } else {
            try_break_rectangles_diagonal(input, &solution, &mut rng)
        };
        let init_rectangles = skip_none!(init_rectangles);

        if solution.rectangles.len() != 0 && solution.rectangles.len() == init_rectangles.len() {
            continue;
        }

        let state = random_greedy(input, &init_rectangles, &mut rng);

        // スコア計算
        let new_score = state.calc_normalized_score(input);
        let score_diff = new_score - current_score;

        if score_diff >= 0 || rng.gen_bool(f64::exp(score_diff as f64 / temp)) {
            // 解の更新
            current_score = new_score;
            accepted_count += 1;
            solution = state;

            if chmax!(best_score, current_score) {
                best_solution = solution.clone();
                update_count += 1;
                not_improved = 0;
            } else {
                not_improved += 1;

                if not_improved >= NOT_IMPROVED_THRESHOLD {
                    solution = best_solution.clone();
                    current_score = best_score;
                    not_improved = 0;
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
) -> Option<Vec<[Vec2; 4]>> {
    let x0 = rng.gen_range(0, input.n - 1);
    let x1 = rng.gen_range(x0 + 1, input.n);
    let y0 = rng.gen_range(0, input.n - 1);
    let y1 = rng.gen_range(y0 + 1, input.n);
    let count = solution.board.get_range_popcnt(x0, y0, x1, y1);

    if (solution.rectangles.len() != 0 && count == 0) || count >= 50 {
        return None;
    }

    let mut init_rectangles = Vec::with_capacity(solution.rectangles.len());
    for rect in solution.rectangles.iter() {
        let p = rect[0];
        let x = p.x as usize;
        let y = p.y as usize;

        if !(x0 <= x && x < x1 && y0 <= y && y < y1) {
            init_rectangles.push(rect.clone());
        }
    }

    Some(init_rectangles)
}

fn try_break_rectangles_diagonal(
    input: &Input,
    solution: &State,
    rng: &mut rand_pcg::Pcg64Mcg,
) -> Option<Vec<[Vec2; 4]>> {
    let dir = rng.gen_range(0, 4) * 2 + 1;
    let x = rng.gen_range(0, input.n) as i32;
    let y = rng.gen_range(0, input.n) as i32;
    let width = rng.gen_range(0, input.n / 2) as i32;
    let height = width;
    let unit_u = UNITS[dir];
    let unit_v = UNITS[rot_c(dir)];

    let p0 = Vec2::new(x, y);
    let p1 = p0 + unit_u * width;
    let p2 = p1 + unit_v * height;
    let p3 = p0 + unit_v * height;

    fn between(p: Vec2, p0: Vec2, p1: Vec2, p2: Vec2) -> bool {
        let p = p - p0;
        let p1 = p1 - p0;
        let p2 = p2 - p0;
        p1.cross(p) <= 0 && p2.cross(p) >= 0
    }

    let mut init_rectangles = Vec::with_capacity(solution.rectangles.len());
    for rect in solution.rectangles.iter() {
        let p = rect[0];

        if !between(p, p0, p1, p3) || !between(p, p2, p3, p1) {
            init_rectangles.push(rect.clone());
        }
    }

    let removed = solution.rectangles.len() - init_rectangles.len();

    if (solution.rectangles.len() != 0 && init_rectangles.len() == 0) || removed >= 50 {
        None
    } else {
        Some(init_rectangles)
    }
}

fn random_greedy(input: &Input, init_rectangles: &[[Vec2; 4]], rng: &mut Pcg64Mcg) -> State {
    let mut state = State::init(input);
    state.rectangles.reserve(init_rectangles.len() * 3 / 2);

    for rect in init_rectangles {
        if rect[1..].iter().all(|p| state.board.is_occupied(*p)) {
            state.apply(input, rect);
        }
    }

    let mut sampler_small = WeightedSampler::<[Vec2; 4]>::new(32);
    let mut sampler = WeightedSampler::<[Vec2; 4]>::new(32);
    let mut next_p = [None; DIR_COUNT];

    for &p2 in state.points.iter() {
        for (dir, next) in next_p.iter_mut().enumerate() {
            *next = state.board.find_next(p2, dir);
        }

        for dir in 0..8 {
            let p1 = skip_none!(next_p[dir]);
            let p3 = skip_none!(next_p[rot_c(dir)]);
            let p0 = p1 + (p3 - p2);

            try_add_candidate(
                input,
                &state,
                p0,
                p1,
                p2,
                p3,
                &mut sampler_small,
                &mut sampler,
            )
        }
    }

    loop {
        let rectangle = if sampler_small.len() > 0 {
            sampler_small.sample(rng)
        } else if sampler.len() > 0 {
            sampler.sample(rng)
        } else {
            break;
        };

        if !state.can_apply(&rectangle) {
            continue;
        }

        state.apply(input, &rectangle);

        for (dir, next) in next_p.iter_mut().enumerate() {
            *next = state.board.find_next(rectangle[0], dir);
        }

        let p1 = rectangle[0];

        for dir in 0..8 {
            let p2 = skip_none!(next_p[dir]);
            let p3 = skip_none!(state.board.find_next(p2, rot_c(dir)));
            let p0 = p1 + (p3 - p2);

            try_add_candidate(
                input,
                &state,
                p0,
                p1,
                p2,
                p3,
                &mut sampler_small,
                &mut sampler,
            )
        }

        let p2 = rectangle[0];

        for dir in 0..8 {
            let p1 = skip_none!(next_p[dir]);
            let p3 = skip_none!(next_p[rot_cc(dir)]);
            let p0 = p1 + (p3 - p2);

            try_add_candidate(
                input,
                &state,
                p0,
                p1,
                p2,
                p3,
                &mut sampler_small,
                &mut sampler,
            )
        }

        let p3 = rectangle[0];

        for dir in 0..8 {
            let p2 = skip_none!(next_p[dir]);
            let p1 = skip_none!(state.board.find_next(p2, rot_cc(dir)));
            let p0 = p1 + (p3 - p2);

            try_add_candidate(
                input,
                &state,
                p0,
                p1,
                p2,
                p3,
                &mut sampler_small,
                &mut sampler,
            )
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
    sampler_small: &mut WeightedSampler<[Vec2; 4]>,
    sampler: &mut WeightedSampler<[Vec2; 4]>,
) {
    if !p0.in_map(input.n)
        || state.board.is_occupied(p0)
        || !state.board.can_connect(p1, p0)
        || !state.board.can_connect(p3, p0)
    {
        return;
    }

    let weight = input.get_weight(p0) as f64;
    let v0 = p1 - p0;
    let v1 = p3 - p0;
    let rectangle = [p0, p1, p2, p3];

    let norm0 = v0.norm2_sq();
    let norm1 = v1.norm2_sq();

    if (norm0 == 1 && norm1 == 1) || (norm0 == 2 && norm1 == 2) {
        sampler_small.push(rectangle, weight);
    } else {
        let weight = weight / (norm0 + norm1) as f64;
        sampler.push(rectangle, weight);
    }
}

#[allow(dead_code)]
mod vector {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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

        #[inline]
        pub fn rot(&self, dir: usize, n: usize) -> Self {
            let mut v = *self;
            let n = n as i32;

            // 180°回転
            if ((dir >> 2) & 1) > 0 {
                v.x = n - 1 - v.x;
                v.y = n - 1 - v.y;
            }

            // 90°回転
            if ((dir >> 1) & 1) > 0 {
                let x = v.y;
                let y = n - 1 - v.x;
                v.x = x;
                v.y = y;
            }

            // 45°回転
            if (dir & 1) > 0 {
                let x = (v.x + v.y) >> 1;
                let y = n - 1 - v.x + v.y;
                v.x = x;
                v.y = y;
            }

            v
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
        }
    }
}

mod sample {
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::acl::fenwicktree::FenwickTree;

    pub struct WeightedSampler<T> {
        n: usize,
        prob: FenwickTree<f64>,
        values: Vec<T>,
    }

    impl<T> WeightedSampler<T> {
        pub fn new(n: usize) -> Self {
            let prob = FenwickTree::new(n, 0.0);
            let values = vec![];

            Self { n, prob, values }
        }

        pub fn len(&self) -> usize {
            self.values.len()
        }

        pub fn push(&mut self, v: T, prob: f64) {
            let index = self.values.len();

            if index >= self.n {
                // 長さを倍にする
                let mut new_prob = FenwickTree::new(2 * self.n, 0.0);
                let mut sum = 0.0;

                for i in 0..self.n {
                    let s = self.prob.accum(i + 1);
                    new_prob.add(i, s - sum);
                    sum = s;
                }

                self.n *= 2;
                self.prob = new_prob;
            }

            self.values.push(v);
            self.prob.add(index, prob);
        }

        pub fn sample(&mut self, rng: &mut Pcg64Mcg) -> T {
            let v = rng.gen_range(0.0, self.prob.accum(self.n));
            let index = self.prob.binary_search(v).min(self.len() - 1);
            let swap_index = self.values.len() - 1;
            let p0 = self.prob.sum(index, index + 1);
            let p1 = self.prob.sum(swap_index, swap_index + 1);
            self.prob.add(index, p1 - p0);
            self.prob.add(swap_index, -p1);

            self.values.swap_remove(index)
        }
    }
}

mod acl {
    pub mod fenwicktree {
        // Reference: https://en.wikipedia.org/wiki/Fenwick_tree
        pub struct FenwickTree<T> {
            n: usize,
            ary: Vec<T>,
            e: T,
        }

        impl<T: Clone + std::ops::AddAssign<T>> FenwickTree<T> {
            pub fn new(n: usize, e: T) -> Self {
                FenwickTree {
                    n,
                    ary: vec![e.clone(); n],
                    e,
                }
            }

            pub fn accum(&self, mut idx: usize) -> T {
                let mut sum = self.e.clone();
                while idx > 0 {
                    sum += self.ary[idx - 1].clone();
                    idx &= idx - 1;
                }
                sum
            }

            /// performs data[idx] += val;
            pub fn add<U: Clone>(&mut self, mut idx: usize, val: U)
            where
                T: std::ops::AddAssign<U>,
            {
                let n = self.n;
                idx += 1;
                while idx <= n {
                    self.ary[idx - 1] += val.clone();
                    idx += idx & idx.wrapping_neg();
                }
            }

            /// Returns data[l] + ... + data[r - 1].
            pub fn sum(&self, l: usize, r: usize) -> T
            where
                T: std::ops::Sub<Output = T>,
            {
                self.accum(r) - self.accum(l)
            }
        }

        impl<T: PartialOrd + std::ops::SubAssign<T> + Default + Copy> FenwickTree<T> {
            pub fn binary_search(&self, mut sum: T) -> usize {
                let depth = 64 - (self.n as u64).leading_zeros();
                let mut index = 0;

                for i in (0..=depth).rev() {
                    let k = index + (1 << i);
                    if k < self.n && self.ary[k - 1] < sum {
                        sum -= self.ary[k - 1];
                        index = k;
                    }
                }

                index
            }
        }

        #[cfg(test)]
        mod tests {
            use super::*;

            #[test]
            fn fenwick_tree_works() {
                let mut bit = FenwickTree::new(5, 0i64);
                // [1, 2, 3, 4, 5]
                for i in 0..5 {
                    bit.add(i, i as i64 + 1);
                }
                assert_eq!(bit.sum(0, 5), 15);
                assert_eq!(bit.sum(0, 4), 10);
                assert_eq!(bit.sum(1, 3), 5);
            }

            #[test]
            fn binary_search() {
                let mut bit = FenwickTree::new(5, 0);
                for i in 0..5 {
                    bit.add(i, i as i64 + 1);
                }

                assert_eq!(bit.binary_search(1), 0);
                assert_eq!(bit.binary_search(9), 3);
                assert_eq!(bit.binary_search(10), 3);
                assert_eq!(bit.binary_search(11), 4);
            }
        }
    }
}
