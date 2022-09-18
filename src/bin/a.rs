use std::time::Instant;

#[allow(unused_imports)]
use proconio::*;
#[allow(unused_imports)]
use rand::prelude::*;
use vector::Map2d;

use crate::vector::{inv, rot_c, Vec2, UNITS};

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

#[derive(Debug, Clone)]
struct Input {
    n: usize,
    m: usize,
    p: Vec<Vec2>,
    score_coef: f64,
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

        let mut input = Input {
            n,
            m,
            p,
            score_coef,
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

    fn to_state(&self) -> State {
        let mut occupied = Map2d::new(vec![false; self.n * self.n], self.n);
        let edges = Map2d::new(vec![[false; 8]; self.n * self.n], self.n);

        for p in self.p.iter() {
            occupied[p] = true;
        }

        State {
            points: self.p.clone(),
            occupied,
            edges,
            rectangles: vec![],
        }
    }
}

#[derive(Debug, Clone)]
struct State {
    points: Vec<Vec2>,
    occupied: Map2d<bool>,
    edges: Map2d<[bool; 8]>,
    rectangles: Vec<[Vec2; 4]>,
}

impl State {
    fn from_input(
        points: Vec<Vec2>,
        occupied: Map2d<bool>,
        edges: Map2d<[bool; 8]>,
        rectangles: Vec<[Vec2; 4]>,
    ) -> Self {
        Self {
            points,
            occupied,
            edges,
            rectangles: vec![],
        }
    }

    fn apply(&mut self, rectangle: &[Vec2; 4]) {
        self.points.push(rectangle[0]);
        self.occupied[rectangle[0]] = true;
        self.rectangles.push(rectangle.clone());

        for (i, &from) in rectangle.iter().enumerate() {
            let to = rectangle[(i + 1) % 4];
            let mut v = from;
            let unit = (to - from).unit();
            let dir = unit.to_dir();

            while v != to {
                debug_assert!(!self.edges[v][dir]);
                self.edges[v][dir] = true;
                v += unit;
                debug_assert!(!self.edges[v][inv(dir)]);
                self.edges[v][inv(dir)] = true;
            }
        }
    }

    fn calc_score(&self, input: &Input) -> i32 {
        let mut total_weight = 0;

        for &p in self.points.iter() {
            total_weight += input.get_weight(p);
        }

        (total_weight as f64 * input.score_coef).round() as i32
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
    let output = greedy(&input);
    println!("{}", output);
}

fn greedy(input: &Input) -> Output {
    let mut state = input.to_state();
    let since = Instant::now();

    while (Instant::now() - since).as_millis() < 4900 {
        let mut best_rectangle = None;
        let mut best_weight = 0;

        for x in 0..(input.n as i32) {
            for y in 0..(input.n as i32) {
                let p0 = Vec2::new(x, y);

                if state.occupied[p0] {
                    continue;
                }

                for dir in 0..8 {
                    let p1 = skip_none!(search(input, &state, p0, dir));
                    let p2 = skip_none!(search(input, &state, p0, rot_c(dir)));
                    let p13 = skip_none!(search(input, &state, p1, rot_c(dir)));
                    let p23 = skip_none!(search(input, &state, p2, dir));

                    if p13 != p23 {
                        continue;
                    }

                    let weight = input.get_weight(p0);

                    if chmax!(best_weight, weight) {
                        best_rectangle = Some([p0, p1, p13, p2]);
                    }
                }
            }
        }

        if let Some(rect) = best_rectangle {
            state.apply(&rect);
        } else {
            break;
        }
    }

    state.to_output()
}

fn search(input: &Input, state: &State, start: Vec2, dir: usize) -> Option<Vec2> {
    let delta = UNITS[dir];
    let mut current = start + delta;

    while current.in_map(input.n) {
        if state.edges[current][inv(dir)] {
            return None;
        } else if state.occupied[current] {
            return Some(current);
        }

        current += delta;
    }

    None
}

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

        pub const fn rot_cc(&self) -> Self {
            Self::new(-self.y, self.x)
        }

        pub const fn rot_c(&self) -> Self {
            Self::new(self.y, -self.x)
        }

        pub fn unit(&self) -> Self {
            Self::new(self.x.signum(), self.y.signum())
        }

        pub fn to_dir(&self) -> usize {
            debug_assert!(self.x.abs() + self.y.abs() == 1);
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

    impl std::fmt::Display for Vec2 {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "({}, {})", self.x, self.y)
        }
    }

    impl std::ops::Neg for Vec2 {
        type Output = Vec2;

        fn neg(self) -> Self::Output {
            Vec2::new(-self.x, -self.y)
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

    pub const fn inv(dir: usize) -> usize {
        dir ^ 4
    }

    pub const fn rot_cc(dir: usize) -> usize {
        (dir + 2) % 8
    }

    pub const fn rot_c(dir: usize) -> usize {
        (dir + 6) % 8
    }

    #[derive(Debug, Clone)]
    pub struct Map2d<T> {
        pub width: usize,
        pub height: usize,
        map: Vec<T>,
    }

    impl<T> Map2d<T> {
        pub fn new(map: Vec<T>, width: usize) -> Self {
            let height = map.len() / width;
            debug_assert!(width * height == map.len());
            Self { width, height, map }
        }
    }

    impl<T> std::ops::Index<Vec2> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, v: Vec2) -> &Self::Output {
            let x = v.x as usize;
            let y = v.y as usize;
            debug_assert!(x < self.width && y < self.width);
            &self.map[y * self.width + x]
        }
    }

    impl<T> std::ops::IndexMut<Vec2> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, v: Vec2) -> &mut Self::Output {
            let x = v.x as usize;
            let y = v.y as usize;
            debug_assert!(x < self.width && y < self.width);
            &mut self.map[y * self.width + x]
        }
    }

    impl<T> std::ops::Index<&Vec2> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, v: &Vec2) -> &Self::Output {
            let x = v.x as usize;
            let y = v.y as usize;
            debug_assert!(x < self.width && y < self.width);
            &self.map[y * self.width + x]
        }
    }

    impl<T> std::ops::IndexMut<&Vec2> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, v: &Vec2) -> &mut Self::Output {
            let x = v.x as usize;
            let y = v.y as usize;
            debug_assert!(x < self.width && y < self.width);
            &mut self.map[y * self.width + x]
        }
    }

    impl<T> std::ops::Index<usize> for Map2d<T> {
        type Output = [T];

        #[inline]
        fn index(&self, row: usize) -> &Self::Output {
            let begin = row * self.width;
            let end = begin + self.width;
            &self.map[begin..end]
        }
    }

    impl<T> std::ops::IndexMut<usize> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, row: usize) -> &mut Self::Output {
            let begin = row * self.width;
            let end = begin + self.width;
            &mut self.map[begin..end]
        }
    }
}
