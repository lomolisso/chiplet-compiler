use cc_pipeline::{DagError, PassGraph, Pipeline, PipelineError};
use tests::{Env, Fixture};

#[test]
fn cycle_is_rejected_with_its_members() {
    let passes = vec![
        Fixture::new("a").requires("k.b").provides("k.a").boxed(),
        Fixture::new("b").requires("k.a").provides("k.b").boxed(),
        Fixture::new("c").provides("k.c").boxed(),
    ];
    assert_eq!(
        PassGraph::build(&passes).unwrap_err(),
        DagError::Cycle(vec!["a".into(), "b".into()])
    );
}

#[test]
fn self_loop_is_a_cycle() {
    let passes = vec![Fixture::new("a").requires("k.a").provides("k.a").boxed()];
    assert_eq!(
        PassGraph::build(&passes).unwrap_err(),
        DagError::Cycle(vec!["a".into()])
    );
}

#[test]
fn missing_provider_is_rejected() {
    let passes = vec![Fixture::new("a").requires("k.nobody").provides("k.a").boxed()];
    assert_eq!(
        PassGraph::build(&passes).unwrap_err(),
        DagError::MissingProvider {
            pass: "a".into(),
            kind: "k.nobody".into()
        }
    );
}

#[test]
fn duplicate_provider_is_rejected() {
    let passes = vec![
        Fixture::new("a").provides("k").boxed(),
        Fixture::new("b").provides("k").boxed(),
    ];
    assert!(matches!(
        PassGraph::build(&passes).unwrap_err(),
        DagError::DuplicateProvider { .. }
    ));
}

#[test]
fn bad_names_are_rejected() {
    let passes = vec![Fixture::new("Hello-A").boxed()];
    assert!(matches!(
        PassGraph::build(&passes).unwrap_err(),
        DagError::BadPassName(_)
    ));
    let passes = vec![Fixture::new("a").provides("K/x").boxed()];
    assert!(matches!(
        PassGraph::build(&passes).unwrap_err(),
        DagError::BadKind { .. }
    ));
    let passes = vec![Fixture::new("a").boxed(), Fixture::new("a").boxed()];
    assert!(matches!(
        PassGraph::build(&passes).unwrap_err(),
        DagError::DuplicatePass(_)
    ));
}

#[test]
fn unknown_target_is_rejected() {
    let env = Env::new();
    let cfg = env.config("[pipeline]\ntargets = [\"report\"]\n", &[]);
    let err = Pipeline::new(cc_passes::registry(), cfg).unwrap_err();
    assert!(
        matches!(err, PipelineError::Dag(DagError::UnknownPass { .. })),
        "{err}"
    );
}

#[test]
fn registry_dag_and_order() {
    let g = PassGraph::build(&cc_passes::registry()).unwrap();
    let order: Vec<&str> = g.topo_order().into_iter().map(|i| g.name(i)).collect();
    assert_eq!(
        order,
        [
            "export",
            "hello_a",
            "hello_b",
            "hello_c",
            "import",
            "components",
            "normalize",
            "kernels",
            "kernel_classes",
            "iteration_space",
            "datapaths"
        ]
    );
    assert_eq!(
        g.to_dot(),
        "digraph passes {\n  rankdir=LR;\n  node [shape=box];\n  \"components\";\n  \"datapaths\";\n  \"export\";\n  \"hello_a\";\n  \"hello_b\";\n  \"hello_c\";\n  \"import\";\n  \"iteration_space\";\n  \"kernel_classes\";\n  \"kernels\";\n  \"normalize\";\n  \
         \"components\" -> \"kernel_classes\" [label=\"components\"];\n  \
         \"components\" -> \"normalize\" [label=\"components\"];\n  \"export\" -> \"import\" [label=\"export\"];\n  \
         \"export\" -> \"kernel_classes\" [label=\"export\"];\n  \
         \"hello_a\" -> \"hello_b\" [label=\"hello.a\"];\n  \"import\" -> \"components\" [label=\"import\"];\n  \
         \"iteration_space\" -> \"datapaths\" [label=\"iteration_space\"];\n  \
         \"kernel_classes\" -> \"datapaths\" [label=\"kernel_classes\"];\n  \
         \"kernel_classes\" -> \"iteration_space\" [label=\"kernel_classes\"];\n  \
         \"kernels\" -> \"datapaths\" [label=\"kernels\"];\n  \
         \"kernels\" -> \"iteration_space\" [label=\"kernels\"];\n  \
         \"kernels\" -> \"kernel_classes\" [label=\"kernels\"];\n  \
         \"normalize\" -> \"kernel_classes\" [label=\"normalize\"];\n  \
         \"normalize\" -> \"kernels\" [label=\"normalize\"];\n}\n"
    );
}
