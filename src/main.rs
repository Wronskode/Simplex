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
    let (mut matrix, mut variables, mut hash_map_vars, is_min, original_cost) = 
    match simplexef64::parse_lp_two_phases(&lpfile) {
        Ok((matrix, variables, is_min, hash_map_vars, original_cost)) => (matrix, variables, hash_map_vars, is_min, original_cost),
        Err(e) => {
            // println!("❌ Failed to parse LP file with error: {:?}", e);
            return (StatusCode::BAD_REQUEST, ("Failed to parse LP file ".to_string()+&e).into_response());
        }
    };
    //println!("Parsed LP file successfully");
    let (variables,z) = 
    match simplexef64::solve_system_two_phases(&mut matrix, &mut variables, &mut hash_map_vars, &original_cost, is_min) {
        Ok((variables,  z)) => (variables, z),
        Err(e) => {
            // println!("❌ Failed to solve LP with error: {:?}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, e.into_response());
        }
    };
    // println!("{:?}", Json((variables.clone(), z.clone())));
    // (StatusCode::OK, Json((variables.iter().map(|(a, b, c)| (a, b.to_string(), c)).collect::<Vec<_>>(), z.const_term.numer().to_string()+"/"+&z.const_term.denom().to_string())).into_response())
    //println!("z = {:?}", z);
    // let variables = variables.iter().map(|(a, b)| (a, b.to_f64())).collect::<Vec<_>>();
    (StatusCode::OK, Json((variables, z)).into_response())

}

async fn branch_and_bound(lpfile: String) -> impl IntoResponse {
    let (variables,_, z) = 
    match simplexef64::branch_and_bound(&lpfile) {
        Ok((variables,  x, z)) => (variables, x, z),
        Err(e) => {
            // println!("❌ Failed to solve LP with error: {:?}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, e.into_response());
        }
    };
    // let variables = variables.iter().map(|(a, b)| (a, b.to_f64())).collect::<Vec<_>>();
    (StatusCode::OK, Json((variables, z)).into_response())
}

fn simplexe_cmd(path: &str) {
    let now = std::time::Instant::now();
    //let argv1 = "../warehouse100.lp";
    let file_string = std::fs::read_to_string(path).unwrap();
    let (mut matrix, mut variables, is_min, mut vars_hash_map, original_cost) = match simplexef64::parse_lp_two_phases(&file_string) {
        Ok((matrix, variables, is_min, vars_hash_map, original_cost)) => (matrix, variables, is_min, vars_hash_map, original_cost),
        Err(e) => {
            println!("❌ Failed to parse LP file with error: {:?}", e);
            return;
        }
    };
    let (variables,z) = 
    match simplexef64::solve_system_two_phases(&mut matrix, &mut variables, &mut vars_hash_map, &original_cost, is_min) {
        Ok((variables,  z)) => (variables, z),
        Err(e) => {
            println!("❌ Failed to solve LP with error: {:?}", e);
            return;
        }
    };
    println!("{:?}\nz = {:?}\nTime taken: {:?}", variables, z, now.elapsed());
}

fn main() {
    let argv1 = std::env::args().nth(1);
    match argv1 {
        Some(path) => {
            if path == "server" {
                server();
            }
            else {
                simplexe_cmd(&path);
            }
        }
        None => {
            println!("Please provide a path to the LP file or uses \"server\" to run the server");
        }
    }
}