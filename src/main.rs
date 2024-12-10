use csv::{ReaderBuilder, WriterBuilder};
use rand::{prelude::*, seq::SliceRandom};
use std::error::Error;
use std::collections::HashSet;
use std::io::{self, Write};

#[derive(Debug, Clone)]
struct ColumnData {
    label: String,
    max_axial_load: f64,
    length: f64,
}

#[derive(Debug, Clone)]
struct SectionData {
    name: String,
    axial_capacity: f64,
    weight_per_unit_length: f64,
}

const POPULATION_SIZE: usize = 300;
const NUM_GENERATIONS: usize = 1000;
const MUTATION_RATE: f64 = 0.02;
const TOURNAMENT_SIZE: usize = 5;
const ELITISM_RATE: usize = 5; // Elitism rate, retain top 5 solutions
const CAPACITY_PENALTY_FACTOR: f64 = 1000.0;

fn load_project_data(filename: &str) -> Result<Vec<ColumnData>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(filename)?;
    let mut data = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let label = record[0].to_string();
        let max_axial_load = record[1].parse::<f64>()?;
        let length = record[2].parse::<f64>()?;
        data.push(ColumnData {
            label,
            max_axial_load,
            length,
        });
    }
    Ok(data)
}

fn load_section_data(filename: &str) -> Result<Vec<SectionData>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(filename)?;
    let mut data = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let name = record[0].to_string();
        let axial_capacity = record[1].parse::<f64>()?;
        let weight_per_unit_length = record[2].parse::<f64>()?;
        data.push(SectionData {
            name,
            axial_capacity,
            weight_per_unit_length,
        });
    }
    Ok(data)
}

fn initialize_population(
    project_data: &[ColumnData],
    sections: &[SectionData],
) -> Vec<Vec<usize>> {
    let mut population = Vec::new();
    let mut rng = thread_rng();

    for _ in 0..POPULATION_SIZE {
        let chromosome = project_data
            .iter()
            .map(|column| {
                let valid_sections: Vec<usize> = sections
                    .iter()
                    .enumerate()
                    .filter(|(_, section)| column.max_axial_load <= section.axial_capacity)
                    .map(|(index, _)| index)
                    .collect();

                *valid_sections.choose(&mut rng).expect("No valid sections found")
            })
            .collect();

        population.push(chromosome);
    }
    population
}

fn fitness(
    chromosome: &[usize],
    project_data: &[ColumnData],
    sections: &[SectionData],
    distinct_section_types: usize,
) -> f64 {
    let mut used_types = HashSet::new();
    let mut total_weight = 0.0;
    let mut penalty = 0.0;

    for (i, &section_index) in chromosome.iter().enumerate() {
        let section = &sections[section_index];
        let column = &project_data[i];

        // Check if the section can support the column load
        if column.max_axial_load > section.axial_capacity {
            penalty += (column.max_axial_load - section.axial_capacity) * CAPACITY_PENALTY_FACTOR;
        }

        total_weight += section.weight_per_unit_length * column.length;
        used_types.insert(section_index);
    }

    // Penalize if the chromosome doesn't meet the required number of distinct section types
    let distinct_count = used_types.len();
    if distinct_count != distinct_section_types {
        penalty += (distinct_section_types as f64 - distinct_count as f64).abs() * 1000.0;
    }

    total_weight + penalty
}

fn tournament_selection(
    population: &[Vec<usize>],
    fitness_values: &[f64],
) -> Vec<usize> {
    let mut rng = thread_rng();
    let mut tournament = Vec::new();

    for _ in 0..TOURNAMENT_SIZE {
        let idx = rng.gen_range(0..population.len());
        tournament.push((idx, fitness_values[idx]));
    }

    tournament.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    population[tournament[0].0].clone()
}

fn crossover(parent1: &[usize], parent2: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut rng = thread_rng();
    let point = rng.gen_range(1..parent1.len());

    let child1 = [&parent1[..point], &parent2[point..]].concat();
    let child2 = [&parent2[..point], &parent1[point..]].concat();

    (child1, child2)
}

fn mutate(chromosome: &mut Vec<usize>, section_indices: &[usize]) {
    let mut rng = thread_rng();
    for gene in chromosome.iter_mut() {
        if rng.gen::<f64>() < MUTATION_RATE {
            *gene = *section_indices.choose(&mut rng).unwrap();
        }
    }
}

fn genetic_algorithm(
    project_data: &[ColumnData],
    sections: &[SectionData],
    distinct_section_types: usize,
) -> Vec<usize> {
    let section_indices: Vec<usize> = (0..sections.len()).collect();
    let mut population = initialize_population(project_data, &sections);
    let mut best_solution = population[0].clone();
    let mut best_fitness = f64::INFINITY;

    for generation in 1..=NUM_GENERATIONS {
        let fitness_values: Vec<f64> = population
            .iter()
            .map(|chromosome| fitness(chromosome, project_data, sections, distinct_section_types))
            .collect();

        let (gen_best_idx, &gen_best_fitness) = fitness_values
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        if gen_best_fitness < best_fitness {
            best_fitness = gen_best_fitness;
            best_solution = population[gen_best_idx].clone();
        }

        let mut new_population = Vec::new();

        // Elitism: Keep the top ELITISM_RATE solutions
        for _ in 0..ELITISM_RATE {
            new_population.push(population[fitness_values.iter().position(|&x| x == best_fitness).unwrap()].clone());
        }

        // Fill the remaining population
        while new_population.len() < POPULATION_SIZE {
            let parent1 = tournament_selection(&population, &fitness_values);
            let parent2 = tournament_selection(&population, &fitness_values);
            let (mut child1, mut child2) = crossover(&parent1, &parent2);
            mutate(&mut child1, &section_indices);
            mutate(&mut child2, &section_indices);
            new_population.push(child1);
            new_population.push(child2);
        }

        new_population.truncate(POPULATION_SIZE); // Ensure the population size is exactly POPULATION_SIZE

        population = new_population;

        // Print generation number and fitness/weight of the best solution
        let total_weight = fitness(&best_solution, project_data, sections, distinct_section_types);
        println!("Generation {}: Best Fitness (Weight) = {:.2}", generation, total_weight);
    }

    best_solution
}

fn main() -> Result<(), Box<dyn Error>> {
    let project_data = load_project_data("project_data.csv")?;
    let sections = load_section_data("column_sections.csv")?;

    // Ask the user for the number of distinct section types
    let mut input = String::new();
    print!("Enter the number of distinct section types: ");
    io::stdout().flush()?;
    io::stdin().read_line(&mut input)?;
    let distinct_section_types: usize = input.trim().parse()?;

    let best_solution = genetic_algorithm(&project_data, &sections, distinct_section_types);

    // Save the best solution
    let mut wtr = WriterBuilder::new().from_path("best_solution.csv")?;
    wtr.write_record(&[
        "Column Label",
        "Maximum Axial Load",
        "Length",
        "Assigned Section",
        "Axial Capacity",
        "Weight of Column",
    ])?;

    for (i, &section_idx) in best_solution.iter().enumerate() {
        let section = &sections[section_idx];
        let column = &project_data[i];
        let weight = section.weight_per_unit_length * column.length;

        wtr.write_record(&[
            &column.label,
            &column.max_axial_load.to_string(),
            &column.length.to_string(),
            &section.name,
            &section.axial_capacity.to_string(),
            &weight.to_string(),
        ])?;
    }

    Ok(())
}
