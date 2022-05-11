use crate::bls12_381::*;
use ark_ec::{
    bls12,
    models::{ModelParameters, SWModelParameters},
    short_weierstrass_jacobian::GroupAffine,
    AffineCurve,
    hashing::curve_maps::wb::WBParams,
};
use ark_ff::{MontFp, QuadExt, Field, Zero, BigInt};

pub type G2Affine = bls12::G2Affine<crate::bls12_381::Parameters>;
pub type G2Projective = bls12::G2Projective<crate::bls12_381::Parameters>;

#[derive(Clone, Default, PartialEq, Eq)]
pub struct Parameters;

impl ModelParameters for Parameters {
    type BaseField = Fq2;
    type ScalarField = Fr;

    /// COFACTOR = (x^8 - 4 x^7 + 5 x^6) - (4 x^4 + 6 x^3 - 4 x^2 - 4 x + 13) //
    /// 9
    /// = 305502333931268344200999753193121504214466019254188142667664032982267604182971884026507427359259977847832272839041616661285803823378372096355777062779109
    #[rustfmt::skip]
    const COFACTOR: &'static [u64] = &[
        0xbc69f08f2ee75b35,
        0x84c6a0ea91b35288,
        0x8e2a8e9145ad7689,
        0x986ff031508ffe13,
        0x29c2f178731db956,
        0xd82bf015d1212b02,
        0xec0ec69d7477c1ae,
        0x954cbc06689f6a35,
        0x9894c0adebbf6b4e,
        0x8020005aaa95551
    ];

    /// COFACTOR_INV = COFACTOR^{-1} mod r
    /// 26652489039290660355457965112010883481355318854675681319708643586776743290055
    #[rustfmt::skip]
    const COFACTOR_INV: Fr = MontFp!(
        Fr, 
        "26652489039290660355457965112010883481355318854675681319708643586776743290055"
    );
}

impl SWModelParameters for Parameters {
    /// COEFF_A = [0, 0]
    const COEFF_A: Fq2 = QuadExt!(g1::Parameters::COEFF_A, g1::Parameters::COEFF_A,);

    /// COEFF_B = [4, 4]
    const COEFF_B: Fq2 = QuadExt!(g1::Parameters::COEFF_B, g1::Parameters::COEFF_B,);

    /// AFFINE_GENERATOR_COEFFS = (G2_GENERATOR_X, G2_GENERATOR_Y)
    const AFFINE_GENERATOR_COEFFS: (Self::BaseField, Self::BaseField) =
        (G2_GENERATOR_X, G2_GENERATOR_Y);

    #[inline(always)]
    fn mul_by_a(_: &Self::BaseField) -> Self::BaseField {
        Self::BaseField::zero()
    }

    fn is_in_correct_subgroup_assuming_on_curve(point: &G2Affine) -> bool {
        // Algorithm from Section 4 of https://eprint.iacr.org/2021/1130.
        //
        // Checks that [p]P = [X]P

        let mut x_times_point = point.mul(BigInt::new([crate::bls12_381::Parameters::X[0], 0, 0, 0]));
        if crate::bls12_381::Parameters::X_IS_NEGATIVE {
            x_times_point = -x_times_point;
        }

        let p_times_point = p_power_endomorphism(point);

        x_times_point.eq(&p_times_point)
    }
}

pub const G2_GENERATOR_X: Fq2 = QuadExt!(G2_GENERATOR_X_C0, G2_GENERATOR_X_C1);
pub const G2_GENERATOR_Y: Fq2 = QuadExt!(G2_GENERATOR_Y_C0, G2_GENERATOR_Y_C1);

/// G2_GENERATOR_X_C0 =
/// 352701069587466618187139116011060144890029952792775240219908644239793785735715026873347600343865175952761926303160
#[rustfmt::skip]
pub const G2_GENERATOR_X_C0: Fq = MontFp!(Fq, "352701069587466618187139116011060144890029952792775240219908644239793785735715026873347600343865175952761926303160");

/// G2_GENERATOR_X_C1 =
/// 3059144344244213709971259814753781636986470325476647558659373206291635324768958432433509563104347017837885763365758
#[rustfmt::skip]
pub const G2_GENERATOR_X_C1: Fq = MontFp!(Fq, "3059144344244213709971259814753781636986470325476647558659373206291635324768958432433509563104347017837885763365758");

/// G2_GENERATOR_Y_C0 =
/// 1985150602287291935568054521177171638300868978215655730859378665066344726373823718423869104263333984641494340347905
#[rustfmt::skip]
pub const G2_GENERATOR_Y_C0: Fq = MontFp!(Fq, "1985150602287291935568054521177171638300868978215655730859378665066344726373823718423869104263333984641494340347905");

/// G2_GENERATOR_Y_C1 =
/// 927553665492332455747201965776037880757740193453592970025027978793976877002675564980949289727957565575433344219582
#[rustfmt::skip]
pub const G2_GENERATOR_Y_C1: Fq = MontFp!(Fq, "927553665492332455747201965776037880757740193453592970025027978793976877002675564980949289727957565575433344219582");

// psi(x,y) = (x**p * PSI_X, y**p * PSI_Y) is the Frobenius composed
// with the quadratic twist and its inverse

// PSI_X = 1/(u+1)^((p-1)/3)
pub const P_POWER_ENDOMORPHISM_COEFF_0 : Fq2 = QuadExt!(
    FQ_ZERO,
    MontFp!(
       Fq,
       "4002409555221667392624310435006688643935503118305586438271171395842971157480381377015405980053539358417135540939437"
    )
);

// PSI_Y = 1/(u+1)^((p-1)/2)
pub const P_POWER_ENDOMORPHISM_COEFF_1: Fq2 = QuadExt!(
    MontFp!(
       Fq,
       "2973677408986561043442465346520108879172042883009249989176415018091420807192182638567116318576472649347015917690530"),
    MontFp!(
       Fq,
       "1028732146235106349975324479215795277384839936929757896155643118032610843298655225875571310552543014690878354869257")
);

pub fn p_power_endomorphism(p: &GroupAffine<Parameters>) -> GroupAffine<Parameters> {
    // The p-power endomorphism for G2 is defined as follows:
    // 1. Note that G2 is defined on curve E': y^2 = x^3 + 4(u+1). To map a point (x, y) in E' to (s, t) in E,
    //    one set s = x / ((u+1) ^ (1/3)), t = y / ((u+1) ^ (1/2)), because E: y^2 = x^3 + 4.
    // 2. Apply the Frobenius endomorphism (s, t) => (s', t'), another point on curve E,
    //    where s' = s^p, t' = t^p.
    // 3. Map the point from E back to E'; that is,
    //    one set x' = s' * ((u+1) ^ (1/3)), y' = t' * ((u+1) ^ (1/2)).
    //
    // To sum up, it maps
    // (x,y) -> (x^p / ((u+1)^((p-1)/3)), y^p / ((u+1)^((p-1)/2)))
    // as implemented in the code as follows.

    let mut res = *p;
    res.x.frobenius_map(1);
    res.y.frobenius_map(1);

    let tmp_x = res.x.clone();
    res.x.c0 = -P_POWER_ENDOMORPHISM_COEFF_0.c1 * &tmp_x.c1;
    res.x.c1 = P_POWER_ENDOMORPHISM_COEFF_0.c1 * &tmp_x.c0;
    res.y *= P_POWER_ENDOMORPHISM_COEFF_1;

    res
}

impl WBParams for Parameters
{
    type IsogenousCurve = g2_swu_iso::SwuIsoParameters;

    const PHI_X_NOM: &'static [<Self::IsogenousCurve as ModelParameters>::BaseField] = &[
        QuadExt!(
                   MontFp!(Fq, "889424345604814976315064405719089812568196182208668418962679585805340366775741747653930584250892369786198727235542"), 
                   MontFp!(Fq, "889424345604814976315064405719089812568196182208668418962679585805340366775741747653930584250892369786198727235542")), 
        QuadExt!(
                   MontFp!(Fq, "0"), 
                   MontFp!(Fq, "2668273036814444928945193217157269437704588546626005256888038757416021100327225242961791752752677109358596181706522")), 
        QuadExt!(
                   MontFp!(Fq, "2668273036814444928945193217157269437704588546626005256888038757416021100327225242961791752752677109358596181706526"), 
                   MontFp!(Fq, "1334136518407222464472596608578634718852294273313002628444019378708010550163612621480895876376338554679298090853261")), 
        QuadExt!(
                   MontFp!(Fq, "3557697382419259905260257622876359250272784728834673675850718343221361467102966990615722337003569479144794908942033"), 
                   MontFp!(Fq, "0")),
    ];
    
    const PHI_X_DEN: &'static [<Self::IsogenousCurve as ModelParameters>::BaseField] = &[
        QuadExt!(
                   MontFp!(Fq, "0"), 
                   MontFp!(Fq, "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559715")), 
        QuadExt!(
                   MontFp!(Fq, "12"), 
                   MontFp!(Fq, "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559775")), 
        QuadExt!(
                   MontFp!(Fq, "1"), 
                   MontFp!(Fq, "0")),
    ];
    
    const PHI_Y_NOM: &'static [<Self::IsogenousCurve as ModelParameters>::BaseField] = &[
        QuadExt!(
                   MontFp!(Fq, "741186954670679146929220338099241510473496818507223682468899654837783638979784789711608820209076974821832272696229"), 
                   MontFp!(Fq, "741186954670679146929220338099241510473496818507223682468899654837783638979784789711608820209076974821832272696229")), 
        QuadExt!(
                   MontFp!(Fq, "0"), 
                   MontFp!(Fq, "3112985209616852417102725420016814343988686637730339466369378550318691283715096116788757044878123294251695545324269")), 
        QuadExt!(
                   MontFp!(Fq, "1334136518407222464472596608578634718852294273313002628444019378708010550163612621480895876376338554679298090853263"), 
                   MontFp!(Fq, "2668273036814444928945193217157269437704588546626005256888038757416021100327225242961791752752677109358596181706524")), 
        QuadExt!(
                   MontFp!(Fq, "1185899127473086635086752540958786416757594909611557891950239447740453822367655663538574112334523159714931636314011"), 
                   MontFp!(Fq, "0")),
    ];
    
    const PHI_Y_DEN: &'static [<Self::IsogenousCurve as ModelParameters>::BaseField] = &[
        QuadExt!(
                   MontFp!(Fq, "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559355"), 
                   MontFp!(Fq, "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559355")), 
        QuadExt!(
                   MontFp!(Fq, "0"), 
                   MontFp!(Fq, "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559571")), 
        QuadExt!(
                   MontFp!(Fq, "18"), 
                   MontFp!(Fq, "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559769")), 
        QuadExt!(
                   MontFp!(Fq, "1"), 
                   MontFp!(Fq, "0")),
    ];
}