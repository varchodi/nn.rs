pub mod maths;
pub mod utils;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod cool {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
