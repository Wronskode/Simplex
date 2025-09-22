use axum::{
    extract::{Json, DefaultBodyLimit},
    http::StatusCode,
    routing::{post},
    Router,
    response::IntoResponse,
};

mod simplexef64;

#[tokio::main]
async fn server() {
    let app = Router::new()
        .route("/simplex", post(simplexe))
        .route("/branch_and_bound", post(branch_and_bound))
        .layer(DefaultBodyLimit::max(1024*1024*50));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8888")
        .await
        .unwrap();

    axum::serve(listener, app.into_make_service())
        .await
        .unwrap();
}

async fn simplexe(lpfile: String) -> impl IntoResponse {
    let simplexe_state = match simplexef64::parse_lp_two_phases(&lpfile) {
        Ok(state) => state,
        Err(e) => {
            // println!("❌ Failed to parse LP file with error: {:?}", e);
            return (StatusCode::BAD_REQUEST, ("Failed to parse LP file ".to_string()+&e).into_response());
        }
    };
    let mut matrix = simplexe_state.matrix;
    let mut variables = simplexe_state.vars;
    let mut hash_map_vars = simplexe_state.map;
    let is_min = if simplexe_state.sense == simplexef64::ObjectiveSense::Min { -1.0 } else { 1.0 };
    let (variables,z) = 
    match simplexef64::execute_two_phase_solution(&mut matrix, &mut variables, &mut hash_map_vars, is_min) {
        Ok((variables,  z)) => (variables, z),
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, e.into_response());
        }
    };
    (StatusCode::OK, Json((variables, z)).into_response())

}

async fn branch_and_bound(lpfile: String) -> impl IntoResponse {
    let (variables,z) = 
    match simplexef64::branch_and_bound(&lpfile, true) {
        Ok((variables, z, _)) => (variables, z),
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, e.into_response());
        }
    };
    (StatusCode::OK, Json((variables, z)).into_response())
}

fn branch_and_bound_cmd(lpfile: String, with_two_phases: bool) {
    let now = std::time::Instant::now();
    let (variables, z, explored_nodes) = 
    match simplexef64::branch_and_bound(&lpfile, with_two_phases) {
        Ok((variables, z, explored_nodes)) => (variables, z, explored_nodes),
        Err(e) => {
            println!("❌ Failed to solve LP with error: {:?}", e);
            return;
        }
    };
    let max_var_len = variables.iter().map(|(var, _)| var.len()).max().unwrap_or(0);
    println!("Actual values of the variables:");
    for (var, val) in variables {
        println!("{:width$} = {}", var, val, width = max_var_len);
    }
    println!("Value of objective function: {z}",);
    println!("Number of explored nodes: {explored_nodes}");
    println!("Time taken: {:?}", now.elapsed());
}

fn main() {
    let argv1 = std::env::args().nth(1);
    let argv2 = std::env::args().nth(2);
    match argv1 {
        Some(path) => {
            if path == "server" {
                server();
            }
            else {
                let file_content = std::fs::read_to_string(&path).unwrap();
                match argv2 {
                    Some(phase) if phase == "--no_two_phases" => {
                        branch_and_bound_cmd(file_content, false);
                    }
                    _ => {
                        branch_and_bound_cmd(file_content, true);
                    }
                }
            }
        }
        None => {
            println!("Please provide a path to the LP file or uses \"server\" to run the server");
        }
    }
}