# This script will:
# 1. Connect to REAPER.
# 2. Get the source file of the first item on the first track.
# 3. Use ffmpeg to check if the file is 16kHz, mono, 16-bit PCM WAV.
# 4. If not, convert it to a temporary WAV file.
# 5. Run Senko diarization on the (possibly temporary) file.
# 6. Create a new, colored track for each unique speaker.
# 7. Split the original media item based on diarization segments.
# 8. Move the new segments to their corresponding speaker track.
#
# ARGS:
# --json_input <path>: Skips diarization and loads results from a JSON file.
# --json_output <path>: Runs diarization and saves results to a JSON file.

import os
import json
import senko
import argparse
from pathlib import Path
import sys
import traceback
import reapy
import ffmpeg  # Added for audio probing and conversion
import tempfile # Added for creating a temporary file
import math
import colorsys

def create_color_palette(speaker_count):
    color_palette = [] 
    for m in range(1, speaker_count+1):
        # Calculate HSL values (H is float 0.0 to 1.0)
        hue_value = (m / speaker_count) + 0.5
        S = 0.2
        L = 0.4
        
        # 1. Use colorsys to convert HSL (H, L, S) to RGB (R, G, B) in 0.0-1.0 range
        # Note: colorsys uses HLS, not HSL, so Lightness and Saturation are swapped.
        r_float, g_float, b_float = colorsys.hls_to_rgb(hue_value % 1.0, L, S)
        
        # 2. Convert 0.0-1.0 floats to 0-255 integers using standard rounding
        R = int(round(r_float * 255))
        G = int(round(g_float * 255))
        B = int(round(b_float * 255))
        
        rgb_tuple = (R, G, B)
        color_palette.append(rgb_tuple)
    return color_palette
    print(color_palette)

def run_senko_diarization(diarizer, rpr_project, json_output_path=None):
    """
    Gets, converts (if needed), diarizes, and returns results.
    Optionally saves results to a JSON file.
    """
    temp_file = None # To hold our temporary file object if we create one
    path_to_diarize = None

    try:
        # 1. Get the media file path from REAPER
        if len(rpr_project.tracks) == 0:
            print("Error: Project has no tracks.")
            return None
        
        original_track = rpr_project.tracks[0]
        if len(original_track.items) == 0:
            print(f"Error: Track '{original_track.name}' has no items.")
            return None

        item = original_track.items[0]
        take = item.active_take
        if not take or not take.source:
            print(f"Error: Item {item.id} has no active take or source.")
            return None
            
        input_media_path = take.source.filename
        print(f"Found media file: {input_media_path}")
        
        # 2. Use ffmpeg to check the file format
        print("Probing file format...")
        try:
            probe = ffmpeg.probe(input_media_path)
        except ffmpeg.Error as e:
            print(f"Error probing file with ffmpeg: {e.stderr.decode()}", file=sys.stderr)
            return None

        audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
        if audio_stream is None:
            print("Error: No audio stream found in the file.")
            return None

        sample_rate = audio_stream.get('sample_rate')
        channels = audio_stream.get('channels')
        codec_name = audio_stream.get('codec_name')

        # Desired format: 16kHz, mono, 16-bit PCM WAV
        is_correct_format = (
            sample_rate == '16000' and
            channels == 1 and
            codec_name == 'pcm_s16le'
        )

        # 3. Convert if necessary
        if is_correct_format:
            print("File is already in the required 16kHz, mono, 16-bit WAV format.")
            path_to_diarize = input_media_path
        else:
            print(f"Converting file (format: {codec_name}, {sample_rate}Hz, {channels}ch) ...")
            # Create a temporary WAV file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            path_to_diarize = temp_file.name
            temp_file.close() # Close the file so ffmpeg can write to it

            try:
                (
                    ffmpeg
                    .input(input_media_path)
                    .output(path_to_diarize, ar=16000, ac=1, codec='pcm_s16le')
                    .run(quiet=True, overwrite_output=True)
                )
                print(f"Converted file saved to temporary path: {path_to_diarize}")
            except ffmpeg.Error as e:
                print(f"Error during ffmpeg conversion: {e.stderr.decode()}", file=sys.stderr)
                return None

        # 4. Run diarization on the correct file
        print(f"Diarizing '{path_to_diarize}'...")
        result_data = diarizer.diarize(path_to_diarize, generate_colors=True)

        # 5. Save to JSON if requested
        if json_output_path:
            results_output = { 
              "merged_segments": result_data["merged_segments"], 
              "speaker_color_sets" : result_data["speaker_color_sets"]
            }
            print(f"Saving results to {json_output_path}...")
            try:
                with open(json_output_path, 'w') as f:
                    json.dump(results_output, f, indent=4)
                print("Successfully saved JSON file.")
            except Exception as e:
                print(f"Error saving JSON file: {e}")
                traceback.print_exc() # Just print error, don't stop

        return result_data # Return the data

    except senko.AudioFormatError:
        # Error message already printed by diarizer
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        # Print the full error details
        traceback.print_exc()
        return None
    finally:
        # 6. Clean up the temporary file if one was created
        if temp_file and os.path.exists(path_to_diarize):
            try:
                os.remove(path_to_diarize)
                print(f"Temporary file '{path_to_diarize}' deleted.")
            except OSError as e:
                print(f"Warning: Could not delete temporary file '{path_to_diarize}': {e}")

def process_results_in_reaper(result_data, rpr_project):
    """
    Takes diarization results (from Senko or JSON) and applies them to REAPER.
    """
    
    if result_data is None or not result_data.get("merged_segments"): # No speakers detected or empty segments
        print("No speaker segments detected in results. Nothing to do.")
        return

    try:
        # 1. Get the media item from REAPER (needed for splitting)
        if len(rpr_project.tracks) == 0:
            print("Error: Project has no tracks.")
            return
        
        original_track = rpr_project.tracks[0]
        if len(original_track.items) == 0:
            print(f"Error: Track '{original_track.name}' has no items.")
            return

        item = original_track.items[0]
        take = item.active_take
        if not take or not take.source:
            print(f"Error: Item {item.id} has no active take or source.")
            return
            
        # Get item properties needed for splitting and moving
        item_start_pos = item.position
        item_end_pos = item_start_pos + item.length
        take_offset = take.start_offset
        
        # 2. Use the provided result_data
        merged_segments = result_data["merged_segments"]

        print(f"Found {len(merged_segments)} segments. Processing in REAPER...")

        # 3. Create new tracks for each speaker
        rpr_project.begin_undo_block()
        try:
            speaker_track_map = {}
            unique_speakers = sorted(list(set(s['speaker'] for s in merged_segments)))
            color_palette = create_color_palette(len(unique_speakers))
            
            print(f"Creating tracks for {len(unique_speakers)} speakers: {', '.join(unique_speakers)}")
            for index, speaker_name in enumerate(unique_speakers):
                new_track = rpr_project.add_track(index+1)
                new_track.name = speaker_name
                new_track.color = color_palette[index]
                
                speaker_track_map[speaker_name] = new_track

            # 4. Split and move segments
            print("Splitting original item and moving segments...")
            item_to_split = original_track.items[0]

            num_segments = len(merged_segments)

            for i, s in enumerate(merged_segments):
                speaker_name = s["speaker"]
                target_track = speaker_track_map[speaker_name]
                
                # Calculate absolute project times for segment
                seg_start_in_file = s["start"]
                seg_end_in_file = s["end"]

                is_last_segment = (i == num_segments - 1)

                if is_last_segment:
                    # It's the last segment, so its end is the split time
                    split_time_in_file = s["end"]
                else:
                    # It's not the last segment, so the split time is the start of the next one
                    next_s = merged_segments[i+1]
                    split_time_in_file = next_s["start"]
                
                item_relative_split = (split_time_in_file - take_offset) 

                if item_relative_split < 0:
                    print(f"Info: Split {i} at {item_relative_split:.2f}s occurs before the item start. Skipping.")
                    continue

                abs_split_time = item_start_pos + item_relative_split

                if abs_split_time > item_end_pos:
                    print(f"Info: Split {i} at {abs_split_time:2f}s occurs after the item ends. Skipping.")
                    continue
                
                if not item_to_split:
                    print(f"Warning: Could not find item on original track for segment at {abs_split_time:.2f}s. Skipping.")
                    continue

                if not is_last_segment and abs_split_time < (item_to_split.position + item_to_split.length):
                    print(f"Splitting segment {i} normally.")
                    segment_item, item_to_split = item_to_split.split(abs_split_time)
                else:
                    print(f"Splitting last segment.")
                    segment_item = item_to_split
                    item_to_split = None # No more items left to split
                
                # Move the final, correctly-sized segment item
                segment_item.track = target_track
                reapy.reascript_api.ThemeLayout_RefreshAll()

            print(f"Successfully processed {len(merged_segments)} segments.")
        
        except Exception as e:
            print(f"Error during REAPER processing: {str(e)}")
            traceback.print_exc()
            rpr_project.end_undo_block(description="Diarize Speakers to Tracks (Failed)")
        else:
            rpr_project.end_undo_block(description="Diarize Speakers to Tracks")

    except Exception as e:
        print(f"Error during REAPER processing: {str(e)}")
        # Print the full error details
        traceback.print_exc()
    # No finally block here, temp file is handled in run_senko_diarization


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Senko REAPER Diarization Script')
    parser.add_argument('--device',
                        choices=['auto', 'cuda', 'coreml', 'cpu'],
                        default='auto',
                        help='Torch device to use for processing (default: auto)')
    parser.add_argument('--json_input',
                        type=str,
                        default=None,
                        help='Path to a .json file to load diarization results from.')
    parser.add_argument('--json_output',
                        type=str,
                        default=None,
                        help='Path to save the diarization results to as a .json file.')
    args = parser.parse_args()

    # Check if reapy_boost is connected to REAPER
    rpr_project = None
    if not reapy.is_inside_reaper():
        # Attempt to connect to a running REAPER instance
        try:
            rpr_project = reapy.Project() # This will throw an error if it can't connect
            print("Connected to REAPER instance.")
        except Exception as e:
            print(f"Error: Could not connect to REAPER. Please ensure REAPER is running.")
            print(f"Details: {e}")
            sys.exit(1)
    else:
        print("Error: This script will not run inside REAPER due to its reliance on external python packages.")
        sys.exit(1)

    if rpr_project is None:
        print("Error: Could not get REAPER project instance.")
        sys.exit(1)

    result_data = None # Initialize

    if args.json_input:
        # --- JSON Input Path ---
        print(f"Loading results from {args.json_input}...")
        try:
            with open(args.json_input, 'r') as f:
                result_data = json.load(f)
            print("Successfully loaded JSON data.")
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    else:
        # --- Full Diarization Path ---
        print("Initializing Senko diarizer...")
        diarizer = senko.Diarizer(device=args.device, warmup=False, quiet=False)
        print("Diarizer warmed up and ready!\n")
        
        result_data = run_senko_diarization(diarizer, rpr_project, args.json_output)
    
    # --- Common Processing Path ---
    if result_data:
        print("Processing results in REAPER...")
        process_results_in_reaper(result_data, rpr_project)
    else:
        print("No diarization data generated or loaded. Exiting.")

    print("\nScript finished.")
