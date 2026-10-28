//! The engine's `points::point_layer` is create-vectors' `point_belongs` mapping: for every layer
//! count and `Pt N: Use` value, the one layer `point_layer` names is exactly the set of layers the
//! glue's predicate accepts.

use create_vectors::glue::point_belongs;
use retime_engine::points::point_layer;

#[test]
fn point_layer_names_the_layer_point_belongs_accepts() {
    for nfg in 0..=3usize {
        let layer_count = nfg + 1;
        for v in -2i64..=8 {
            let owners: Vec<usize> = (0..layer_count).filter(|&l| point_belongs(v, l, layer_count)).collect();
            assert_eq!(point_layer(v, nfg).into_iter().collect::<Vec<_>>(), owners, "nfg {nfg}, value {v}");
        }
    }
}
