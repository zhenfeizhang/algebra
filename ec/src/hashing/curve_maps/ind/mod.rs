use crate::ark_std::borrow::ToOwned;
use crate::ark_std::Zero;
use crate::{short_weierstrass::SWCurveConfig, AffineRepr};
use ark_ff::PrimeField;
use sha2::Digest;
use sha2::Sha512;

pub trait IndHashConfig: SWCurveConfig
where
    Self::BaseField: PrimeField,
{
    // m = (q - 10) // 27
    const M: Self::BaseField;
    // w = b^((q-1) // 3)
    const W: Self::BaseField;
    // z = w.nth_root(3)
    const Z: Self::BaseField;
    // c1 = (b/z).nth_root(3)
    const C: Self::BaseField;
    // sb = b.nth_root(2)
    const SB: Self::BaseField;

    /// affine curve point
    type GroupAffine: AffineRepr;

    /// map an element in Fq^2 to Group
    fn hash_to_curve<B: AsRef<[u8]>>(input: B) -> Self::GroupAffine {
        Self::hash_to_curve_unchecked(input).clear_cofactor()
    }

    /// Map an element in Fq^2 to Curve without clearing cofactor.
    fn hash_to_curve_unchecked<B: AsRef<[u8]>>(input: B) -> Self::GroupAffine {
        let t = Self::eta(input);
        let nums = Self::phi(&t[0], &t[1]);
        let p = Self::h_prime(&[nums[0], nums[1], nums[2], nums[3], t[0], t[1]]);
        if nums[4] == Self::BaseField::zero() {
            Self::GroupAffine::zero()
        } else if nums[3] == Self::BaseField::zero() {
            Self::GroupAffine::generator()
        } else {
            p
        }
    }

    /// rational map Fq^2 -> T(Fq)
    /// returns nums0, nums1, nums2, den, s1s2
    /// rational map Fq^2 -> T(Fq)
    //  [1, Lemma 1] states that T is given in the affine space A^5(y0,y1,y2,t1,t2) by the two equations
    //  y1^2 - b = b*(y0^2 - b)*t1^3,
    //  y2^2 - b = b^2*(y0^2 - b)*t2^3,
    //  where tj := xj/x0.
    //  The threefold T can be regarded as an elliptic curve in A^3(y0,y1,y2) over the function field F := Fq(s1,s2),
    //  where sj := tj^3.
    //  By virtue of [1, Theorem 2] the non-torsion part of the Mordell-Weil group T(F) is generated by phi from [1, Theorem 1].
    fn phi(t1: &Self::BaseField, t2: &Self::BaseField) -> [Self::BaseField; 5] {
        // constants
        let one = Self::BaseField::from(1u64);
        let two = Self::BaseField::from(2u64);
        let three = Self::BaseField::from(3u64);

        let s1 = *t1 * *t1 * *t1;
        let s2 = *t2 * *t2 * *t2;
        let s1s1 = s1 * s1;
        let s2s2 = s2 * s2;
        let s1s2 = s1 * s2;

        let c2 = Self::C * Self::C;

        let c3 = Self::C * c2;
        let c4 = c2 * c2;

        //	a20 = c2*s1s1
        let a20 = c2 * s1s1;
        // a11 = 2*c3*s1s2
        let a11 = two * c3 * s1s2;
        // 	a10 = 2*c*s1
        let a10 = two * Self::C * s1;

        // a02 = c4*s2s2
        let a02 = c4 * s2s2;

        // a01 = 2*c2*s2
        let a01 = two * c2 * s2;
        // 	num0 = sb*(a20 - a11 + a10 + a02 + a01 - 3)
        let num0 = Self::SB * (a20 - a11 + a10 + a02 + a01 - three);

        // num1 = sb*(-3*a20 + a11 + a10 + a02 - a01 + 1)
        let num1 = Self::SB * (-three * a20 + a11 + a10 + a02 - a01 + one);
        // num2 = sb*(a20 + a11 - a10 - 3*a02 + a01 + 1)
        let num2 = Self::SB * (a20 + a11 - a10 - three * a02 + a01 + one);
        // den = a20 - a11 - a10 + a02 - a01 + 1
        let den = a20 - a11 - a10 + a02 - a01 + one;

        [num0, num1, num2, den, s1s2]
    }

    /// hash function to the plane Fq^2
    fn eta<B: AsRef<[u8]>>(input: B) -> [Self::BaseField; 2] {
        let mut s0 = input.as_ref().to_owned();
        s0.push('0' as u8);
        let mut s1 = input.as_ref().to_owned();
        s1.push('1' as u8);

        let mut hasher = Sha512::new();
        hasher.update(s0);
        let output = hasher.finalize();
        let t1 = Self::BaseField::from_be_bytes_mod_order(&output);

        let mut hasher = Sha512::new();
        hasher.update(s1);
        let output = hasher.finalize();
        let t2 = Self::BaseField::from_be_bytes_mod_order(&output);

        [t1, t2]
    }

    // auxiliary map from the threefold T to Eb
    fn h_prime(inputs: &[Self::BaseField; 6]) -> Self::GroupAffine;
}
