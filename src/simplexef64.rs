use pest::Parser;
use pest_derive::Parser;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::collections::HashMap;

#[derive(Parser)]
#[grammar = "lexer.pest"]
pub struct LPParser;
const PRECISION: f64 = 1.0e-6;
#[inline(always)]
pub fn solve_system_bigm(
    matrix: &mut [Vec<f64>],
    variables: &mut [Variable],
    vars_hash_map: &mut HashMap<String, usize>,
    is_min: f64,
) -> Result<(Vec<(String, f64)>, f64), String> {
    // rayon::ThreadPoolBuilder::new()
    // .num_threads(4)
    // .build_global()
    // .unwrap();
    let borned = big_m(matrix, variables, vars_hash_map, false);
    if !borned {
        return Err("Le problème est non borné".to_string());
    }
    let solved = check_all_constraints(matrix, variables);
    if !solved {
        return Err("Le problème est infaisable".to_string());
    }
    get_solution(matrix, variables, vars_hash_map, is_min)
}

#[inline(always)]
pub fn solve_system_two_phases(
    matrix: &mut [Vec<f64>],
    variables: &mut [Variable],
    vars_hash_map: &mut HashMap<String, usize>,
    original_cost: &HashMap<String, f64>,
    is_min: f64,
) -> Result<(Vec<(String, f64)>, f64), String> {
    two_phases(
        matrix,
        variables,
        vars_hash_map,
        original_cost,
        false,
        is_min,
    );
    let solved = check_all_constraints(matrix, variables);
    if !solved {
        return Err("Le problème est infaisable".to_string());
    }
    get_solution(matrix, variables, vars_hash_map, is_min)
}

#[inline(always)]
fn get_solution(
    matrix: &[Vec<f64>],
    variables: &[Variable],
    vars_hash_map: &HashMap<String, usize>,
    is_min: f64,
) -> Result<(Vec<(String, f64)>, f64), String> {
    let mut vars_string = Vec::with_capacity(variables.len());
    let mut z = 0.0;
    for (index, var) in variables.iter().enumerate() {
        if var.is_slack || var.is_artificial {
            continue;
        }
        vars_string.push((
            vars_hash_map
                .iter()
                .find(|(_, y)| **y == index)
                .unwrap()
                .0
                .to_string(),
            if var.in_base {
                matrix[var.ligne][0]
            } else {
                0.0
            },
        ));
        if var.in_base {
            z += var.cout_original * matrix[var.ligne][0];
        }
    }
    let z = is_min * z;
    Ok((vars_string, z))
}

#[inline(always)]
fn get_objective(matrix: &[Vec<f64>], variables: &[Variable], is_min: f64) -> f64 {
    let mut z = 0.0;
    for var in variables.iter() {
        if var.in_base {
            z += var.cout_original * matrix[var.ligne][0];
        }
    }
    z * is_min
}

#[inline(always)]
fn check_all_constraints(matrix: &[Vec<f64>], variables: &[Variable]) -> bool {
    let mut vars_sorted_by_column = variables.iter().collect::<Vec<_>>();
    vars_sorted_by_column.sort_by(|a, b| a.column.cmp(&b.column));
    for ligne in matrix.iter() {
        let val = ligne[0];
        let scalar = scalar_product(
            &ligne.iter().skip(1).copied().collect::<Vec<_>>(),
            &vars_sorted_by_column
                .iter()
                .map(|x| {
                    if !x.in_base {
                        0.0
                    } else {
                        matrix[x.ligne][0]
                            * (if x.in_base && !x.is_artificial {
                                1.0
                            } else {
                                0.0
                            })
                    }
                })
                .collect::<Vec<_>>(),
        );
        if (val - scalar).abs() > PRECISION {
            return false;
        }
    }
    true
}

#[inline(always)]
pub fn parse_lp_bigm(
    filename: &str,
) -> Result<(Vec<Vec<f64>>, Vec<Variable>, f64, HashMap<String, usize>, bool), String> {
    let file = match LPParser::parse(Rule::program, filename) {
        Ok(mut file) => file.next().unwrap(),
        Err(e) => {
            return Err(format!("Error parsing file: {}", e));
        }
    };
    let mut matrix = vec![];
    let mut variables = HashMap::new();
    let mut count_slack = 0;
    let mut count_artificial = 0;
    let mut current_col = 1;
    let mut is_min = 1.0;
    let mut current_row = 0;
    let mut var_list = vec![];
    let mut bnb = false;
    for line in file.into_inner() {
        match line.as_rule() {
            Rule::function => {
                let mut cost = 0.0;
                for token in line.into_inner() {
                    match token.as_rule() {
                        Rule::obj => {
                            if token.as_str() == "min" {
                                is_min = -1.0;
                            }
                        }
                        Rule::coeff => {
                            let coeff_str = token.as_str();
                            let (sign, mut coeff_str) = match coeff_str {
                                _ if coeff_str.starts_with('+') => (1.0, coeff_str[1..].trim()),
                                _ if coeff_str.starts_with('-') => (-1.0, coeff_str[1..].trim()),
                                _ => {
                                    if coeff_str.chars().count() == 0 {
                                        (1.0, "1")
                                    } else {
                                        (1.0, coeff_str)
                                    }
                                }
                            };
                            if coeff_str.is_empty() {
                                coeff_str = "1";
                            }
                            cost = coeff_str.trim().parse::<f64>().unwrap() * sign * is_min;
                        }
                        Rule::varname => {
                            let var_name = token.as_str().trim();
                            if !variables.contains_key(var_name) {
                                variables.insert(var_name.to_string(), var_list.len());
                                var_list.push(Variable {
                                    in_base: false,
                                    cout_original: cost,
                                    ligne: usize::MAX,
                                    column: current_col,
                                    is_slack: false,
                                    is_artificial: false,
                                    is_integer: false,
                                });
                                current_col += 1;
                            } else {
                                let var = variables.get_mut(var_name).unwrap();
                                var_list[*var].cout_original += cost;
                            }
                        }
                        _ => {}
                    }
                }
            }

            Rule::constraint => {
                let mut constraint_vars: Vec<(usize, f64)> = vec![];
                let mut rhs = 0.0;
                let mut relation: Option<Rule> = None;

                let mut tokens = line.into_inner().peekable();
                while let Some(token) = tokens.next() {
                    match token.as_rule() {
                        Rule::coeff => {
                            let coeff_str = token.as_str().trim();
                            let (sign, mut coeff_str) = match coeff_str {
                                _ if coeff_str.starts_with('+') => (1.0, coeff_str[1..].trim()),
                                _ if coeff_str.starts_with('-') => (-1.0, coeff_str[1..].trim()),
                                _ => {
                                    if coeff_str.chars().count() == 0 {
                                        (1.0, "1")
                                    } else {
                                        (1.0, coeff_str)
                                    }
                                }
                            };
                            if coeff_str.is_empty() {
                                coeff_str = "1";
                            }
                            let coeff = coeff_str.parse::<f64>().unwrap() * sign;
                            if let Some(var_token) = tokens.next() {
                                if var_token.as_rule() == Rule::varname {
                                    let var_name = var_token.as_str().trim();
                                    let column = if let Some(var) = variables.get(var_name) {
                                        var_list[*var].column
                                    } else {
                                        variables.insert(var_name.to_string(), var_list.len());
                                        var_list.push(Variable {
                                            in_base: false,
                                            cout_original: 0.0,
                                            ligne: usize::MAX,
                                            column: current_col,
                                            is_slack: false,
                                            is_artificial: false,
                                            is_integer: false,
                                        });
                                        current_col += 1;
                                        current_col - 1
                                    };
                                    constraint_vars.push((column, coeff));
                                }
                            }
                        }
                        Rule::leq | Rule::geq | Rule::eq => {
                            relation = Some(token.as_rule());
                        }
                        Rule::number => {
                            rhs = token.as_str().parse::<f64>().unwrap();
                        }
                        _ => {}
                    }
                }
                let mut row = vec![0.0];
                for (col, coeff) in constraint_vars {
                    if col >= row.len() {
                        row.resize(col + 1, 0.0);
                    }
                    row[col] = coeff;
                }
                match relation {
                    Some(Rule::leq) => {
                        let slack_name = format!("s{}", count_slack);
                        let slack_col = current_col;
                        if slack_col >= row.len() {
                            row.resize(slack_col + 1, 0.0);
                        }
                        row[slack_col] = 1.0;
                        variables.insert(slack_name, var_list.len());
                        var_list.push(Variable {
                            in_base: true,
                            cout_original: 0.0,
                            ligne: current_row,
                            column: slack_col,
                            is_slack: true,
                            is_artificial: false,
                            is_integer: false,
                        });
                        current_row += 1;
                        count_slack += 1;
                        current_col += 1;
                    }
                    Some(Rule::geq) => {
                        let slack_name = format!("s{}", count_slack);
                        let art_name = format!("a{}", count_artificial);

                        let slack_col = current_col;
                        if slack_col >= row.len() {
                            row.resize(slack_col + 1, 0.0);
                        }
                        row[slack_col] = -1.0;
                        variables.insert(slack_name, var_list.len());
                        var_list.push(Variable {
                            in_base: false,
                            cout_original: 0.0,
                            ligne: usize::MAX,
                            column: slack_col,
                            is_slack: true,
                            is_artificial: false,
                            is_integer: false,
                        });
                        count_slack += 1;
                        current_col += 1;

                        let art_col = current_col;
                        if art_col >= row.len() {
                            row.resize(art_col + 1, 0.0);
                        }
                        row[art_col] = 1.0;
                        variables.insert(art_name, var_list.len());
                        var_list.push(Variable {
                            in_base: true,
                            cout_original: -1.0e12,
                            ligne: current_row,
                            column: art_col,
                            is_slack: false,
                            is_artificial: true,
                            is_integer: false,
                        });
                        current_row += 1;
                        count_artificial += 1;
                        current_col += 1;
                    }
                    Some(Rule::eq) => {
                        let art_name = format!("a{}", count_artificial);
                        let art_col = current_col;
                        if art_col >= row.len() {
                            row.resize(art_col + 1, 0.0);
                        }
                        row[art_col] = 1.0;
                        variables.insert(art_name, var_list.len());
                        var_list.push(Variable {
                            in_base: true,
                            cout_original: -1.0e12,
                            ligne: current_row,
                            column: art_col,
                            is_slack: false,
                            is_artificial: true,
                            is_integer: false,
                        });
                        current_row += 1;
                        count_artificial += 1;
                        current_col += 1;
                    }
                    _ => {}
                }
                row[0] = rhs;
                matrix.push(row);
            }
            Rule::ilp => {
                bnb = true;
                let var_name = line.into_inner().next().unwrap().as_str().trim();
                if let Some(var) = variables.get(var_name) {
                    var_list[*var].is_integer = true;
                }
            }
            _ => {
                println!("Unknown rule: {:?}", line.as_rule());
            }
        }
    }
    let max_len = matrix.iter().map(|row| row.len()).max().unwrap_or(0);
    for row in &mut matrix {
        if row.len() < max_len {
            row.resize(max_len, 0.0);
        }
    }
    Ok((matrix, var_list, is_min, variables, bnb))
}

#[inline(always)]
pub fn parse_lp_two_phases(
    file_content: &str,
) -> Result<
    (
        Vec<Vec<f64>>,
        Vec<Variable>,
        f64,
        HashMap<String, usize>,
        HashMap<String, f64>,
        bool,
    ),
    String,
> {
    let file = match LPParser::parse(Rule::program, file_content) {
        Ok(mut file) => file.next().unwrap(),
        Err(e) => {
            return Err(format!("Error parsing file: {}", e));
        }
    };
    let mut matrix = vec![];
    let mut variables = HashMap::new();
    let mut count_slack = 0;
    let mut count_artificial = 0;
    let mut current_col = 1;
    let mut is_min = 1.0;
    let mut current_row = 0;
    let mut var_list = vec![];
    let mut orignal_cost = HashMap::new();
    let mut bnb = false;
    for line in file.into_inner() {
        match line.as_rule() {
            Rule::function => {
                let mut cost = 0.0;
                for token in line.into_inner() {
                    match token.as_rule() {
                        Rule::obj => {
                            if token.as_str() == "min" {
                                is_min = -1.0;
                            }
                        }
                        Rule::coeff => {
                            let coeff_str = token.as_str();
                            let (sign, mut coeff_str) = match coeff_str {
                                _ if coeff_str.starts_with('+') => (1.0, coeff_str[1..].trim()),
                                _ if coeff_str.starts_with('-') => (-1.0, coeff_str[1..].trim()),
                                _ => {
                                    if coeff_str.chars().count() == 0 {
                                        (1.0, "1")
                                    } else {
                                        (1.0, coeff_str)
                                    }
                                }
                            };
                            if coeff_str.is_empty() {
                                coeff_str = "1";
                            }
                            cost = coeff_str.trim().parse::<f64>().unwrap() * sign * is_min;
                        }
                        Rule::varname => {
                            let var_name = token.as_str().trim();
                            if !variables.contains_key(var_name) {
                                variables.insert(var_name.to_string(), var_list.len());
                                var_list.push(Variable {
                                    in_base: false,
                                    cout_original: 0.0,
                                    ligne: usize::MAX,
                                    column: current_col,
                                    is_slack: false,
                                    is_artificial: false,
                                    is_integer: false,
                                });
                                orignal_cost.insert(var_name.to_string(), cost);
                                current_col += 1;
                            } else {
                                let var = variables.get_mut(var_name).unwrap();
                                var_list[*var].cout_original += cost;
                                orignal_cost
                                    .insert(var_name.to_string(), var_list[*var].cout_original);
                            }
                        }
                        _ => {}
                    }
                }
            }

            Rule::constraint => {
                let mut constraint_vars: Vec<(usize, f64)> = vec![];
                let mut rhs = 0.0;
                let mut relation: Option<Rule> = None;

                let mut tokens = line.into_inner().peekable();
                while let Some(token) = tokens.next() {
                    match token.as_rule() {
                        Rule::coeff => {
                            let coeff_str = token.as_str().trim();
                            let (sign, mut coeff_str) = match coeff_str {
                                _ if coeff_str.starts_with('+') => (1.0, coeff_str[1..].trim()),
                                _ if coeff_str.starts_with('-') => (-1.0, coeff_str[1..].trim()),
                                _ => {
                                    if coeff_str.chars().count() == 0 {
                                        (1.0, "1")
                                    } else {
                                        (1.0, coeff_str)
                                    }
                                }
                            };
                            if coeff_str.is_empty() {
                                coeff_str = "1";
                            }
                            let coeff = coeff_str.parse::<f64>().unwrap() * sign;
                            if let Some(var_token) = tokens.next() {
                                if var_token.as_rule() == Rule::varname {
                                    let var_name = var_token.as_str().trim();
                                    let column = if let Some(var) = variables.get(var_name) {
                                        var_list[*var].column
                                    } else {
                                        variables.insert(var_name.to_string(), var_list.len());
                                        var_list.push(Variable {
                                            in_base: false,
                                            cout_original: 0.0,
                                            ligne: usize::MAX,
                                            column: current_col,
                                            is_slack: false,
                                            is_artificial: false,
                                            is_integer: false,
                                        });
                                        orignal_cost.insert(var_name.to_string(), 0.0);
                                        current_col += 1;
                                        current_col - 1
                                    };
                                    constraint_vars.push((column, coeff));
                                }
                            }
                        }
                        Rule::leq | Rule::geq | Rule::eq => {
                            relation = Some(token.as_rule());
                        }
                        Rule::number => {
                            rhs = token.as_str().parse::<f64>().unwrap();
                        }
                        _ => {}
                    }
                }
                let mut row = vec![0.0];
                for (col, coeff) in constraint_vars {
                    if col >= row.len() {
                        row.resize(col + 1, 0.0);
                    }
                    row[col] = coeff;
                }
                match relation {
                    Some(Rule::leq) => {
                        let slack_name = format!("s{}", count_slack);
                        let slack_col = current_col;
                        if slack_col >= row.len() {
                            row.resize(slack_col + 1, 0.0);
                        }
                        row[slack_col] = 1.0;
                        variables.insert(slack_name.to_string(), var_list.len());
                        var_list.push(Variable {
                            in_base: true,
                            cout_original: 0.0,
                            ligne: current_row,
                            column: slack_col,
                            is_slack: true,
                            is_artificial: false,
                            is_integer: false,
                        });
                        orignal_cost.insert(slack_name, 0.0);
                        current_row += 1;
                        count_slack += 1;
                        current_col += 1;
                    }
                    Some(Rule::geq) => {
                        let slack_name = format!("s{}", count_slack);
                        let art_name = format!("a{}", count_artificial);

                        let slack_col = current_col;
                        if slack_col >= row.len() {
                            row.resize(slack_col + 1, 0.0);
                        }
                        row[slack_col] = -1.0;
                        variables.insert(slack_name.to_string(), var_list.len());
                        var_list.push(Variable {
                            in_base: false,
                            cout_original: 0.0,
                            ligne: usize::MAX,
                            column: slack_col,
                            is_slack: true,
                            is_artificial: false,
                            is_integer: false,
                        });
                        orignal_cost.insert(slack_name, 0.0);
                        count_slack += 1;
                        current_col += 1;

                        let art_col = current_col;
                        if art_col >= row.len() {
                            row.resize(art_col + 1, 0.0);
                        }
                        row[art_col] = 1.0;
                        variables.insert(art_name.to_string(), var_list.len());
                        var_list.push(Variable {
                            in_base: true,
                            cout_original: -1.0,
                            ligne: current_row,
                            column: art_col,
                            is_slack: false,
                            is_artificial: true,
                            is_integer: false,
                        });
                        orignal_cost.insert(art_name, 0.0);
                        current_row += 1;
                        count_artificial += 1;
                        current_col += 1;
                    }
                    Some(Rule::eq) => {
                        let art_name = format!("a{}", count_artificial);
                        let art_col = current_col;
                        if art_col >= row.len() {
                            row.resize(art_col + 1, 0.0);
                        }
                        row[art_col] = 1.0;
                        variables.insert(art_name.to_string(), var_list.len());
                        var_list.push(Variable {
                            in_base: true,
                            cout_original: -1.0,
                            ligne: current_row,
                            column: art_col,
                            is_slack: false,
                            is_artificial: true,
                            is_integer: false,
                        });
                        orignal_cost.insert(art_name, 0.0);
                        current_row += 1;
                        count_artificial += 1;
                        current_col += 1;
                    }
                    _ => {}
                }
                row[0] = rhs;
                matrix.push(row);
            }
            Rule::ilp => {
                bnb = true;
                let var_name = line.into_inner().next().unwrap().as_str().trim();
                if let Some(var) = variables.get(var_name) {
                    var_list[*var].is_integer = true;
                }
            }
            _ => {
                println!("Unknown rule: {:?}", line.as_rule());
            }
        }
    }
    let max_len = matrix.iter().map(|row| row.len()).max().unwrap_or(0);
    for row in &mut matrix {
        if row.len() < max_len {
            row.resize(max_len, 0.0);
        }
    }
    Ok((matrix, var_list, is_min, variables, orignal_cost, bnb))
}


// Simplex iteration
#[inline(always)]
fn update_array(
    matrix: &mut [Vec<f64>],
    variables: &mut [Variable],
    sorted_by_column: &mut [usize],
    in_base: &mut [f64],
    in_phase_one: bool,
    in_phase_two: bool,
) -> (bool, bool) {
    let (min_col_index, min) = sorted_by_column.par_iter().skip(1).enumerate().map(|(i, _)| {
        let y = variables[sorted_by_column[i]];
        if !y.in_base && (!in_phase_two || !y.is_artificial) {
            Some((i+1, scalar_product_column(in_base, matrix, i+1) - y.cout_original))
        }
        else {
            None
        }
    }).filter_map(|x| x).min_by(|a, b| a.1.total_cmp(&b.1)).unwrap_or((0, f64::MAX));
    if min >= 0.0 {
        return (true, true);
    }
    let mut min = f64::MAX;
    let mut line_index = 0;
    let mut leaved = false;
    for (i, item) in matrix.iter().enumerate() {
        if item[min_col_index] <= 0.0 {
            continue;
        }
        let scalar = item[0] / item[min_col_index];
        if in_phase_one
            && scalar == min
            && variables
                .iter()
                .find(|v| v.ligne == i)
                .unwrap()
                .is_artificial
        {
            line_index = i;
        }
        if scalar < min && scalar >= 0.0 {
            leaved = true;
            min = scalar;
            line_index = i;
        }
    }
    if !leaved {
        return (true, false);
    }
    if let Some(var) = variables.iter().position(|v| v.ligne == line_index) {
        variables[var].in_base = false;
        sorted_by_column[variables[var].column - 1] = var;
        variables[var].ligne = usize::MAX;
    }
    if let Some(y) = sorted_by_column.get_mut(min_col_index - 1) {
        let v = variables.get_mut(*y).unwrap();
        v.in_base = true;
        v.ligne = line_index;
        in_base[line_index] = v.cout_original;
        sorted_by_column[min_col_index - 1] = *y;
        let pos = variables
            .iter()
            .position(|v| v.column == min_col_index)
            .unwrap();
        variables[pos].in_base = true;
        variables[pos].ligne = line_index;
    }
    let pivot = matrix[line_index][min_col_index];
    matrix[line_index].iter_mut().for_each(|x| {
            if (*x).abs() > PRECISION {
                *x /= pivot;
            }
            else {
                *x = 0.0;
            }
    });
    let pivot_row = matrix[line_index].clone();
    matrix.par_iter_mut().enumerate().filter(|(i, row)| *i != line_index && row[min_col_index].abs() > PRECISION).for_each(|(_, row)| {
        let coeff = row[min_col_index];
        pivot_row.iter().enumerate().filter(|(_, x)| (**x).abs() > PRECISION).for_each(|(j, x)| {
                row[j] -= x * coeff;
        });
    });
    (false, true)
}

// fn scalar_product(x: &[f64], y: &[f64]) -> f64 {
//     let mut z = 0.0;
//     for i in 0..x.len() {
//         z += x[i]*y[i];
//     }
//     z
// }

// #[inline(always)]
// fn scalar_product_column(x: &[f64], matrix: &[Vec<f64>], j: usize) -> f64 {
//     let mut sum = 0.0;
//     x.iter()
//         .enumerate()
//         .filter(|(i, xi)| **xi != 0.0 && matrix[*i][j] != 0.0)
//         .for_each(|(i, xi)| {
//             sum += xi * matrix[i][j];
//         });
//     sum
// }

#[inline(always)]
fn scalar_product(x: &[f64], y: &[f64]) -> f64 {
    x.par_iter()
        .zip(y.par_iter())
        .filter(|(xi, yi)| (**xi).abs() > PRECISION  && (**yi).abs() > PRECISION )
        .map(|(xi, yi)| xi * yi)
        .sum()
}

fn scalar_product_column(x: &[f64], matrix: &[Vec<f64>], j: usize) -> f64 {
    x.par_iter()
        .enumerate()
        .filter(|(i, xi)| (**xi).abs() > PRECISION && matrix[*i][j].abs() > PRECISION)
        .map(|(i, &xi)| xi * matrix[i][j])
        .sum()
}
#[inline(always)]
fn big_m(
    matrix: &mut [Vec<f64>],
    variables: &mut [Variable],
    hmap_vars: &mut HashMap<String, usize>,
    print: bool,
) -> bool {
    let mut sorted_by_column = hmap_vars.values().copied().collect::<Vec<_>>();
    sorted_by_column.sort_by(|a, b| variables[*a].column.cmp(&variables[*b].column));
    let mut in_base = variables
        .par_iter()
        .filter(|x| x.in_base)
        .map(|x| (x.ligne, x.cout_original))
        .collect::<Vec<_>>();
    in_base.sort_by(|a, b| a.0.cmp(&b.0));
    let mut in_base = in_base.iter().map(|x| x.1).collect::<Vec<_>>();
    loop {
        if print {
            print_system(matrix, variables, hmap_vars, true);
        }
        let (ended, borned) = update_array(
            matrix,
            variables,
            &mut sorted_by_column,
            &mut in_base,
            false,
            false,
        );
        if print {
            print_system(matrix, variables, hmap_vars, true);
        }
        if ended {
            if print {
                print_system(matrix, variables, hmap_vars, true);
            }
            return borned;
        }
    }
}

#[inline(always)]
fn two_phases(
    matrix: &mut [Vec<f64>],
    variables: &mut [Variable],
    hmap_vars: &mut HashMap<String, usize>,
    original_cost: &HashMap<String, f64>,
    print: bool,
    is_min: f64,
) -> bool {
    let mut sorted_by_column = hmap_vars.values().copied().collect::<Vec<_>>();
    sorted_by_column.sort_by(|a, b| variables[*a].column.cmp(&variables[*b].column));
    let mut in_base = variables
        .par_iter()
        .filter(|x| x.in_base)
        .map(|x| (x.ligne, x.cout_original))
        .collect::<Vec<_>>();
    in_base.sort_by(|a, b| a.0.cmp(&b.0));
    let mut in_base = in_base.iter().map(|x| x.1).collect::<Vec<_>>();
    if print {
        print_system(matrix, variables, hmap_vars, true);
    }
    loop {
        let (s1, s2) = update_array(
            matrix,
            variables,
            &mut sorted_by_column,
            &mut in_base,
            true,
            false,
        );
        if print {
            print_system(matrix, variables, hmap_vars, true);
        }
        let z = get_objective(matrix, variables, is_min);
        let all_positive = s1 && s2;
        matrix.par_iter_mut().for_each(|row| {
            row.iter_mut().for_each(|x| {
                if (*x).abs() <= PRECISION {
                    *x = 0.0;
                }
                });
            });
        if z.abs() < PRECISION {
            let art_in_base = variables
                .iter()
                .filter(|x| x.in_base && x.is_artificial && matrix[x.ligne][0].abs() > PRECISION);
            if art_in_base.count() > 0 {
                return false;
            }
            let column_vars_hashmap = hmap_vars
                .par_iter()
                .map(|(_, y)| {
                    (
                        variables[*y].column,
                        hmap_vars.iter().find(|(_, z)| **z == *y).unwrap().0,
                    )
                })
                .collect::<HashMap<_, _>>();
            for x in variables.iter_mut() {
                let v = column_vars_hashmap.get(&x.column).unwrap();
                let original_cost = *original_cost.get(&v.to_string()).unwrap();
                if x.in_base {
                    in_base[x.ligne] = original_cost;
                }
                x.cout_original = original_cost;
            }
            loop {
                let (ended, borned) = update_array(
                    matrix,
                    variables,
                    &mut sorted_by_column,
                    &mut in_base,
                    false,
                    true,
                );
                matrix.par_iter_mut().for_each(|row| {
                        row.iter_mut().for_each(|x| {
                            if (*x).abs() <= PRECISION {
                                *x = 0.0;
                            }
                        });
                    });
                if print {
                    print_system(matrix, variables, hmap_vars, true);
                }
                if ended {
                    return borned;
                }
            }
        } else if all_positive {
            return false;
        }
    }
}
#[derive(Clone)]
struct Node {
    base_lp: String,
    constraints: Vec<(String, String, f64)>,
}

impl Node {
    fn to_lp_string(&self) -> String {
        let mut full_lp = String::with_capacity(self.base_lp.len() + self.constraints.len() * 30);
        full_lp.push_str(&self.base_lp);
        for (var, op, val) in &self.constraints {
            full_lp.push_str(&format!("\n{} {} {};", var, op, val));
        }
        full_lp
    }
}

fn is_integer_solution(vars_string: &Vec<(String, f64)>, vars_hash_map: &std::collections::HashMap<String, usize>, variables: &Vec<Variable>) -> bool {
    for (name, v) in vars_string {
        if let Some(idx) = vars_hash_map.get(name) {
            let var = &variables[*idx];
            if !var.is_slack && !var.is_artificial && var.is_integer && (v - v.round()).abs() > PRECISION {
                return false;
            }
        }
    }
    true
}

fn update_best_solution(best_solution: &mut Option<(Vec<(String, f64)>, f64)>, vars_string: &Vec<(String, f64)>, z: f64, is_min: f64) {
    let is_better = match best_solution {
        None => true,
        Some((_, best_z)) => (is_min == -1.0 && z < *best_z) || (is_min == 1.0 && z > *best_z),
    };
    if is_better {
        *best_solution = Some((vars_string.clone(), z));
    }
}

pub fn branch_and_bound(file: &str) -> Result<(Vec<(String, f64)>, f64, f64), String> {
    let mut stack = vec![Node {
        base_lp: file.to_string(),
        constraints: vec![],
    }];
    let mut best_solution: Option<(Vec<(String, f64)>, f64)> = None;
    let mut is_min_bnb = 1.0;

    while let Some(node) = stack.pop() {
        let lp_str = node.to_lp_string();
        let (mut matrix, mut variables, is_min, mut vars_hash_map, original_cost, bnb) =
            match parse_lp_two_phases(&lp_str) {
                Ok(v) => v,
                Err(e) => {
                    println!("parse_lp_two_phases failed for:\n{}\nError: {:?}", lp_str, e);
                    continue;
                }
            };
        is_min_bnb = is_min;
        let (vars_string, z) = match solve_system_two_phases(
            &mut matrix,
            &mut variables,
            &mut vars_hash_map,
            &original_cost,
            is_min,
        ) {
            Ok(v) => v,
            Err(_) => continue,
        };

        if !bnb {
            return Ok((vars_string, z, is_min));
        }

        let is_current_better = match &best_solution {
            None => true,
            Some((_, best_z)) => (is_min == -1.0 && z < *best_z) || (is_min == 1.0 && z > *best_z),
        };

        if !is_current_better {
             continue; // Élague si la solution courante est moins bonne que la meilleure
        }

        if is_integer_solution(&vars_string, &vars_hash_map, &variables) {
            update_best_solution(&mut best_solution, &vars_string, z, is_min);
            continue; // Élague, car on a trouvé une solution entière
        }
        
        let mut most_fractional: Option<(String, f64)> = None;
        let mut max_fractionality = 0.0;
        
        for (name, v) in &vars_string {
            if let Some(idx) = vars_hash_map.get(name) {
                let var = &variables[*idx];
                if !var.is_slack && !var.is_artificial && var.is_integer {
                    let frac = (v - v.round()).abs();
                    if frac > PRECISION && frac > max_fractionality {
                        max_fractionality = frac;
                        most_fractional = Some((name.clone(), *v));
                    }
                }
            }
        }

        if let Some((nom, val)) = most_fractional {
            let value_inf = val.floor();
            let value_sup = val.ceil();
            let mut constraints1 = node.constraints.clone();
            constraints1.push((nom.clone(), "<=".to_string(), value_inf));
            stack.push(Node {
                base_lp: node.base_lp.clone(),
                constraints: constraints1,
            });

            let mut constraints2 = node.constraints;
            constraints2.push((nom, ">=".to_string(), value_sup));
            stack.push(Node {
                base_lp: node.base_lp,
                constraints: constraints2,
            });
        }
    }

    match best_solution {
        Some((vars, z)) => Ok((vars, z, is_min_bnb)),
        None => Err("Pas de solution entière trouvée".to_string()),
    }
}

fn print_system(
    matrix: &[Vec<f64>],
    variables: &[Variable],
    hash_map_vars: &HashMap<String, usize>,
    show_z_row: bool,
) {
    let variables = hash_map_vars
        .iter()
        .map(|(x, y)| (x.to_string(), variables[*y]))
        .collect::<HashMap<_, _>>();
    for x in variables.iter() {
        println!("{:?} {:?}", x.1.in_base, x.0);
    }
    let mut vars_sorted = BTreeMap::new();
    for var in variables.iter() {
        vars_sorted.insert(var.1.column, var.0.clone());
    }
    print!("{:<8}", "Base");
    print!("{:<8}", "Coût");
    print!("{:<8}", "b");
    for name in vars_sorted.values() {
        print!("{:<8}", name);
    }
    println!();
    let mut base_vars = variables
        .iter()
        .filter(|(_, v)| v.in_base)
        .collect::<Vec<_>>();
    base_vars.sort_by_key(|(_, v)| v.ligne);

    for (i, ligne) in matrix.iter().enumerate() {
        let base = base_vars
            .iter()
            .find(|v| v.1.ligne == i)
            .map(|(x, _)| x.as_str())
            .unwrap_or("-");
        print!("{:<8}", base);
        print!("{}    ", variables.get(base).unwrap().cout_original);
        for val in ligne {
            print!("{:<8}", val);
        }
        println!();
    }
    if show_z_row {
        let mut couts_z = vec!["Z".to_string()];
        couts_z.push("".to_string()); // colonne b
        for var in vars_sorted.values() {
            let mut cb = variables
                .iter()
                .filter(|v| v.1.in_base)
                .map(|v| v.1)
                .collect::<Vec<_>>();
            let my_var = variables.get(var).unwrap();
            cb.sort_by(|a, b| a.ligne.cmp(&b.ligne));
            let cb = cb
                .iter()
                .filter(|x| x.in_base)
                .map(|x| x.cout_original)
                .collect::<Vec<_>>();
            let cout = my_var.cout_original;
            let column: Vec<f64> = matrix.iter().map(|x| x[my_var.column]).collect();
            let scalar = scalar_product(&cb, &column) - cout;
            couts_z.push(format!("{}", scalar));
        }
        for val in couts_z {
            print!("{:<8}", val);
        }
        println!();
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Variable {
    in_base: bool,
    cout_original: f64,
    ligne: usize,
    column: usize,
    is_slack: bool,
    is_artificial: bool,
    is_integer: bool,
}