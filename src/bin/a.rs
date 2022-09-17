#[allow(unused_imports)]
use proconio::*;
#[allow(unused_imports)]
use rand::prelude::*;

use crate::vector::Vec2;

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

#[derive(Debug, Clone)]
struct Input {}

fn main() {
    let v = Vec2::new(0, 0);
    eprintln!("{}", -v);
    todo!();
}

mod vector {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Vec2 {
        x: i32,
        y: i32,
    }

    impl Vec2 {
        pub const fn new(x: i32, y: i32) -> Self {
            Self { x, y }
        }

        pub fn in_map(&self, n: usize) -> bool {
            let n = n as u32;
            (self.x as u32) < n && (self.y as u32) < n
        }

        pub const fn rot90(&self) -> Self {
            Self::new(-self.y, self.x)
        }

        pub fn unit(&self) -> Self {
            Self::new(self.x.signum(), self.y.signum())
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
