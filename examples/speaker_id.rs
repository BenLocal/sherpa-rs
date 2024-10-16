/*
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_speakerverification_speakernet.onnx
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/biden.wav -O biden.wav
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/obama.wav -O obama.wav
cargo run --example speaker_id
*/
use eyre::{bail, Result};
use sherpa_rs::{embedding_manager, speaker_id};
use std::collections::HashMap;

fn main() -> Result<()> {
    // Define paths to the audio files
    let audio_files = vec!["obama.wav", "biden.wav"];

    let config = speaker_id::ExtractorConfig {
        model: "nemo_en_speakerverification_speakernet.onnx".into(),
        ..Default::default()
    };
    let mut extractor = speaker_id::EmbeddingExtractor::new(config)?;

    // Read and process each audio file, compute embeddings
    let mut embeddings = Vec::new();
    for file in &audio_files {
        let mut reader = hound::WavReader::open(file)?;
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect();
        let sample_rate = reader.spec().sample_rate as i32;
        if sample_rate != 16000 {
            bail!("The sample rate must be 16000.");
        }
        let embedding = extractor.compute_speaker_embedding(sample_rate, samples)?;
        embeddings.push((file.to_string(), embedding));
    }

    // Create the embedding manager
    let mut embedding_manager =
        embedding_manager::EmbeddingManager::new(extractor.embedding_size.try_into().unwrap()); // Assuming dimension 512 for embeddings

    // Map to store speakers and their corresponding files
    let mut speaker_map: HashMap<String, Vec<String>> = HashMap::new();
    let mut speaker_counter = 0;

    // Process each embedding and identify speakers
    for (file, embedding) in &embeddings {
        if let Some(speaker_name) = embedding_manager.search(embedding, 0.5) {
            // Add file to existing speaker
            speaker_map
                .entry(speaker_name)
                .or_default()
                .push(file.clone());
        } else {
            // Register a new speaker and add the embedding
            embedding_manager.add(
                format!("speaker {}", speaker_counter),
                &mut embedding.clone(),
            )?;
            speaker_map
                .entry(format!("speaker {}", speaker_counter))
                .or_default()
                .push(file.clone());
            speaker_counter += 1;
        }
    }

    // Print results
    println!("--------");
    println!("📊 Speaker Identification Summary:");
    for (speaker_id, files) in &speaker_map {
        println!("Speaker {}: {:?}", speaker_id, files);
    }

    Ok(())
}
