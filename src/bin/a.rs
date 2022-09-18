use std::time::Instant;

use bitboard::Board;
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

mod bitboard {
    use crate::vector::{Vec2, DIR_COUNT, UNITS};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    struct Bitset {
        v: u64,
    }

    impl Bitset {
        #[inline]
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
        edges: [Vec<Bitset>; DIR_COUNT],
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
            let edges = points.clone();

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
                let has_edge = self.edges[dir][v_rot.y as usize]
                    .contains_range(v_rot.x as u32, (v_rot.x + d) as u32);

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
            let (dir, y, x1, x2) = self.get_rot(v1, v2);
            let has_point = self.points[dir][y].contains_range(x1 + 1, x2);
            let has_edge = self.points[dir][y].contains_range(x1, x2);
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
            self.connect_inner(v2, v1);
        }

        #[inline]
        pub fn disconnect(&mut self, v1: Vec2, v2: Vec2) {
            self.disconnect_inner(v1, v2);
            self.disconnect_inner(v2, v1);
        }

        #[inline]
        fn connect_inner(&mut self, v1: Vec2, v2: Vec2) {
            let (dir, y, x1, x2) = self.get_rot(v1, v2);
            self.edges[dir][y].set_range(x1, x2);
        }

        #[inline]
        fn disconnect_inner(&mut self, v1: Vec2, v2: Vec2) {
            let (dir, y, x1, x2) = self.get_rot(v1, v2);
            self.edges[dir][y].unset_range(x1, x2);
        }

        #[inline]
        fn get_rot(&self, v1: Vec2, v2: Vec2) -> (usize, usize, u32, u32) {
            let dir = (v2 - v1).unit().to_dir();
            let v1_rot = v1.rot(dir, self.n);
            let v2_rot = v2.rot(dir, self.n);

            debug_assert!(v1_rot.y == v2_rot.y);
            let y = v1_rot.y as usize;
            let x1 = v1_rot.x as u32;
            let x2 = v2_rot.x as u32;

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
}

#[derive(Debug, Clone)]
struct State {
    points: Vec<Vec2>,
    board: Board,
    rectangles: Vec<[Vec2; 4]>,
}

impl State {
    fn init(input: &Input) -> Self {
        let points = input.p.clone();
        let board = Board::init(input.n, &input.p);
        let rectangles = vec![];

        Self {
            points,
            board,
            rectangles,
        }
    }

    fn apply(&mut self, rectangle: &[Vec2; 4]) {
        self.points.push(rectangle[0]);
        self.board.add_point(rectangle[0]);
        self.rectangles.push(rectangle.clone());

        for (i, &from) in rectangle.iter().enumerate() {
            let to = rectangle[(i + 1) % 4];
            self.board.connect(from, to);
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
    let mut state = State::init(input);
    let since = Instant::now();

    while (Instant::now() - since).as_millis() < 4900 {
        let mut best_rectangle = None;
        let mut best_weight = 0;

        for x in 0..(input.n as i32) {
            for y in 0..(input.n as i32) {
                let p0 = Vec2::new(x, y);

                if state.board.is_occupied(p0) {
                    continue;
                }

                for dir in 0..8 {
                    let p1 = skip_none!(state.board.find_next(p0, dir));
                    let p2 = skip_none!(state.board.find_next(p0, rot_c(dir)));
                    let p13 = skip_none!(state.board.find_next(p1, rot_c(dir)));
                    let p23 = skip_none!(state.board.find_next(p2, dir));

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
