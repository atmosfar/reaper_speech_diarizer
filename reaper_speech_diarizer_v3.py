import os
import senko
import argparse
import sys
import traceback
import reapy
import ffmpeg
import tempfile
import colorsys
import itertools
import numpy as np
import scipy.signal
import scipy.io.wavfile
import operator

# --- Configuration ---
DEFAULT_IGNORE_TRACKS = {"music", "fx", "sfx", "atmos", "bgm"}
RMS_THRESHOLD = 0.002    # -54dB
PNR_THRESHOLD = 4.0      # Minimum Peak-to-Noise Ratio
BANDPASS_LOW = 300       # Hz
BANDPASS_HIGH = 4000     # Hz

def create_color_palette(count):
    color_palette = [] 
    for m in range(1, count+1):
        hue_value = (m / count) + 0.5
        S = 0.2
        L = 0.4
        r_float, g_float, b_float = colorsys.hls_to_rgb(hue_value % 1.0, L, S)
        R = int(round(r_float * 255))
        G = int(round(g_float * 255))
        B = int(round(b_float * 255))
        color_palette.append((R, G, B))
    return color_palette

def get_all_project_items(project):
    all_items = []
    for track in project.tracks:
        all_items.extend(track.items)
    return all_items

def get_items_to_process(project, force_all=False):
    if force_all:
        print("Flag --all detected: Processing entire project.")
        return get_all_project_items(project)

    selected_items = list(project.selected_items)
    if selected_items:
        print(f"Selection Mode: Processing {len(selected_items)} explicitly selected items.")
        return selected_items

    selected_tracks = list(project.selected_tracks)
    if selected_tracks:
        print(f"Selection Mode: Processing {len(selected_tracks)} selected tracks.")
        items = []
        for track in selected_tracks:
            items.extend(track.items)
        return items

    print("Selection Mode: Processing entire project (no selection found).")
    return get_all_project_items(project)

def filter_ignored_tracks(items, ignore_set):
    valid_items = []
    ignored_counts = {}
    for item in items:
        try:
            track_name = item.track.name.strip().lower()
        except AttributeError:
            continue 
        if track_name in ignore_set:
            ignored_counts[track_name] = ignored_counts.get(track_name, 0) + 1
            continue
        valid_items.append(item)
    if ignored_counts:
        print(f"Ignored items on tracks: {ignored_counts}")
    return valid_items

def build_source_map(items):
    source_map = {}
    for item in items:
        take = item.active_take
        if not take or not take.source: continue
        filename = take.source.filename
        if filename not in source_map: source_map[filename] = []
        source_map[filename].append(item)
    return source_map

def get_or_create_speaker_track(project, track_name, track_color):
    for track in project.tracks:
        if track.name == track_name: return track
    new_track = project.add_track()
    new_track.name = track_name
    new_track.color = track_color
    return new_track

def load_audio_chunk_raw(path, start=None, duration=None, sr=16000):
    try:
        stream = ffmpeg.input(path)
        if start is not None and duration is not None:
            if start < 0:
                duration += start 
                start = 0
            if duration <= 0: return np.array([])
            stream = ffmpeg.input(path, ss=start, t=duration)
        
        out, _ = (
            stream
            .output('pipe:', format='s16le', ac=1, ar=sr)
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        raw = np.frombuffer(out, np.int16)
        if raw.size == 0: return raw

        sig = raw.astype(np.float64)
        sig = sig - np.mean(sig)
        sig = sig / 32768.0 
        return sig
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}", file=sys.stderr)
        return np.array([])

def apply_bandpass(sig, sr=16000):
    if sig.size < 100: return sig
    sos = scipy.signal.butter(5, [BANDPASS_LOW, BANDPASS_HIGH], btype='band', fs=sr, output='sos')
    return scipy.signal.sosfilt(sos, sig)

def find_best_anchor(file_path, segments, debug=False, min_dur=2.0, max_dur=20.0):
    sorted_segments = sorted(segments, key=lambda x: x["end"] - x["start"], reverse=True)
    
    for i, seg in enumerate(sorted_segments[:10]): 
        seg_dur = seg["end"] - seg["start"]
        if seg_dur < min_dur: continue
        
        check_dur = min(seg_dur, max_dur)
        sig = load_audio_chunk_raw(file_path, start=seg["start"], duration=check_dur)
        
        if sig.size == 0: continue
        
        rms = np.sqrt(np.mean(sig**2))
        
        if rms > RMS_THRESHOLD:
            filtered_sig = apply_bandpass(sig)
            norm_sig = filtered_sig / (np.max(np.abs(filtered_sig)) + 1e-6)
            return norm_sig, seg["start"], rms, i
        
    return None, None, None, None

def get_speaker_rms(file_path, segments):
    """Calculates the average RMS of the top 3 longest segments for a speaker."""
    sorted_segments = sorted(segments, key=lambda x: x["end"] - x["start"], reverse=True)
    rms_values = []
    
    for seg in sorted_segments[:3]:
        dur = min(seg["end"] - seg["start"], 5.0) # Check 5s chunks
        sig = load_audio_chunk_raw(file_path, start=seg["start"], duration=dur)
        if sig.size > 0:
            rms_values.append(np.sqrt(np.mean(sig**2)))
            
    if not rms_values: return 0.0
    return np.mean(rms_values)

def calculate_correlation_stats(anchor, target):
    if len(anchor) >= len(target): return 0, 0.0, 0.0
    corr = scipy.signal.correlate(target, anchor, mode='valid', method='fft')
    lag_idx = np.argmax(corr)
    peak_val = corr[lag_idx]
    mean_noise = np.mean(np.abs(corr)) + 1e-9
    pnr = peak_val / mean_noise
    return lag_idx, peak_val, pnr

def get_source_start_time(project, source_filename):
    min_pos = float('inf')
    found = False
    for track in project.tracks:
        for item in track.items:
            take = item.active_take
            if take and take.source and take.source.filename == source_filename:
                if item.position < min_pos:
                    min_pos = item.position
                found = True
    return min_pos if found else 0.0

def shift_items_for_source(project, source_filename, offset_seconds):
    count = 0
    direction_str = "LATER (Right)" if offset_seconds < 0 else "EARLIER (Left)"
    
    print(f"    Action: Shifting {os.path.basename(source_filename)}")
    print(f"            {direction_str} by {abs(offset_seconds):.4f}s")
    
    for track in project.tracks:
        for item in track.items:
            take = item.active_take
            if take and take.source and take.source.filename == source_filename:
                item.position -= offset_seconds
                count += 1
    reapy.reascript_api.ThemeLayout_RefreshAll()

def process_source(diarizer, project, filename, items, source_idx, source_color):
    print(f"\n--- Processing Source {source_idx}: {os.path.basename(filename)} ---")

    temp_file_obj = None 
    path_to_diarize = filename
    used_temp_file = False
    
    try:
        probe = ffmpeg.probe(path_to_diarize)
    except (ffmpeg.Error, KeyError, ValueError) as e:
        print(f"Error probing file: {e}", file=sys.stderr)
        return None, None

    audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
    if audio_stream is None: return None, None

    sample_rate = audio_stream.get('sample_rate')
    channels = audio_stream.get('channels')
    codec_name = audio_stream.get('codec_name')
    is_correct_format = (sample_rate == '16000' and channels == 1 and codec_name == 'pcm_s16le')

    if not is_correct_format:
        print(f"Converting ({codec_name}, {sample_rate}Hz)...")
        temp_file_obj = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        path_to_diarize = temp_file_obj.name
        temp_file_obj.close()
        used_temp_file = True
        try:
            (ffmpeg.input(filename).output(path_to_diarize, ar=16000, ac=1, codec='pcm_s16le')
             .run(quiet=True, overwrite_output=True))
        except ffmpeg.Error as e:
            print(f"Conversion error: {e.stderr.decode()}", file=sys.stderr)
            return None, None

    try:
        result = diarizer.diarize(path_to_diarize)
    except Exception as e:
        print(f"Diarization error: {str(e)}")
        traceback.print_exc()
        return None, path_to_diarize if used_temp_file else None

    segments = result.get("merged_segments", [])
    if segments:
        speaker_track_map = {}
        source_prefix = f"{source_idx:02d}"
        project.begin_undo_block()
        try:
            for item in items:
                item_start = item.position
                item_len = item.length
                take = item.active_take
                take_offset = take.start_offset
                file_window_start = take_offset
                file_window_end = take_offset + item_len
                relevant_segments = [s for s in segments if s['end'] > file_window_start and s['start'] < file_window_end]
                if not relevant_segments: continue

                parent_track_idx = item.track.index + 1
                current_item = item
                for i, seg in enumerate(relevant_segments):
                    speaker_label = seg['speaker']
                    track_name_key = f"T{parent_track_idx:02d}_{source_prefix}_{speaker_label}"
                    if track_name_key not in speaker_track_map:
                        target_track = get_or_create_speaker_track(project, track_name_key, source_color)
                        speaker_track_map[track_name_key] = target_track
                    target_track = speaker_track_map[track_name_key]
                    seg_start_clamped = max(seg['start'], file_window_start)
                    raw_end = relevant_segments[i+1]['start'] if i < len(relevant_segments) - 1 else seg['end']
                    seg_end_clamped = min(raw_end, file_window_end)
                    proj_start = item_start + (seg_start_clamped - take_offset)
                    proj_end = item_start + (seg_end_clamped - take_offset)
                    t = 0.001
                    if proj_start > current_item.position + t:
                        left_item, right_item = current_item.split(proj_start)
                        current_item = right_item
                    if proj_end < current_item.position + current_item.length - t:
                        left_item, right_item = current_item.split(proj_end)
                        left_item.track = target_track
                        current_item = right_item
                    else:
                        current_item.track = target_track
                        break
                    reapy.reascript_api.ThemeLayout_RefreshAll()
            project.end_undo_block(description=f"Diarize Source {source_idx}")
        except Exception as e:
            print(f"Editing error: {e}")
            project.end_undo_block(description=f"Diarize Fail")

    return result, path_to_diarize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['auto', 'cuda', 'coreml', 'cpu'], default='auto')
    parser.add_argument('--align', action='store_true', help='Perform pairwise audio alignment')
    parser.add_argument('--debug', action='store_true', help='Enable verbose debugging and save anchor audio')
    parser.add_argument('--all', action='store_true', help='Ignore UI selection and process all items in project')
    parser.add_argument('--ignore', type=str, help='Comma-separated list of track names to ignore (overrides default)')
    args = parser.parse_args()

    if reapy.is_inside_reaper():
        print("Error: Run from external terminal.")
        sys.exit(1)
    
    try:
        rpr_project = reapy.Project()
        _ = rpr_project.tracks 
    except Exception as e:
        print(f"Error: Could not connect to REAPER. {e}")
        sys.exit(1)

    if args.ignore:
        ignore_set = {x.strip().lower() for x in args.ignore.split(',')}
        print(f"Custom ignore list loaded: {ignore_set}")
    else:
        ignore_set = DEFAULT_IGNORE_TRACKS

    all_proj_items = get_all_project_items(rpr_project)
    items = get_items_to_process(rpr_project, force_all=args.all)
    
    total_count = len(all_proj_items)
    selected_count = len(items)

    if not args.all and (0 < selected_count < total_count):
        print(f"\n[Warning] You have selected {selected_count} items out of {total_count} total in the project.")
        confirm = input("Are you sure you want to process this subset? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("Operation cancelled by user.")
            sys.exit(0)

    valid_items = filter_ignored_tracks(items, ignore_set)
    source_map = build_source_map(valid_items)
    unique_sources = sorted(source_map.keys())
    
    if not unique_sources:
        print("No valid sources.")
        sys.exit(0)

    palette = create_color_palette(len(unique_sources))
    diarizer = senko.Diarizer(device=args.device, warmup=(len(unique_sources)>1), quiet=False)
    
    all_speakers = []
    temp_files_to_clean = set() 
    processing_file_map = {} 

    # 6. Main Loop
    for i, filename in enumerate(unique_sources):
        source_idx = i + 1
        result, processed_path = process_source(diarizer, rpr_project, filename, source_map[filename], source_idx, palette[i])
        
        if processed_path and processed_path != filename:
            temp_files_to_clean.add(processed_path)
        
        if result:
            processing_file_map[filename] = processed_path 
            
            raw_segments = result.get("merged_segments", [])
            file_centroids = result.get("speaker_centroids", {})
            
            # --- Pre-calculate RMS for logic check ---
            spk_rms_map = {}
            if file_centroids:
                for spk_name in file_centroids.keys():
                    spk_segs = [s for s in raw_segments if s['speaker'] == spk_name]
                    rms = get_speaker_rms(processed_path, spk_segs)
                    spk_rms_map[spk_name] = rms

            if file_centroids:
                for speaker_name, centroid_array in file_centroids.items():
                    spk_segments = [s for s in raw_segments if s['speaker'] == speaker_name]
                    all_speakers.append({
                        "filename": filename,
                        "speaker": speaker_name,
                        "centroid": centroid_array,
                        "segments": spk_segments,
                        "rms": spk_rms_map.get(speaker_name, 0.0)
                    })

    # 7. Alignment Phase
    if args.align:
        if len(unique_sources) > 1:
            print("\n--- Starting Pairwise Alignment (RMS Logic Check) ---")
            
            # Group speakers by file
            file_speakers = {}
            for s in all_speakers:
                if s["filename"] not in file_speakers: file_speakers[s["filename"]] = []
                file_speakers[s["filename"]].append(s)

            matches = []
            for spk_a, spk_b in itertools.combinations(all_speakers, 2):
                if spk_a["filename"] == spk_b["filename"]: continue
                similarity = senko.speaker_similarity(spk_a["centroid"], spk_b["centroid"])
                if similarity >= 0.8:
                    matches.append({"score": similarity, "ref": spk_a, "target": spk_b})

            matches.sort(key=operator.itemgetter("score"), reverse=True)
            aligned_files = set() 
            
            if matches:
                rpr_project.begin_undo_block()
                try:
                    for m in matches:
                        ref = m["ref"]
                        target = m["target"]
                        if target["filename"] in aligned_files: continue

                        # --- RMS Logic Check ---
                        # Logic: Check if we are trying to align using a "Silent Bleed" path.
                        # We have matched Ref(S1) to Target(S1).
                        # Let's see who is louder in their respective files.
                        
                        rms_ref = ref["rms"]
                        rms_tgt = target["rms"]
                        
                        # Find the "Other" speakers in these files to calculate ratios
                        ref_others = [s for s in file_speakers[ref["filename"]] if s["speaker"] != ref["speaker"]]
                        tgt_others = [s for s in file_speakers[target["filename"]] if s["speaker"] != target["speaker"]]
                        
                        max_ref_other_rms = max([s["rms"] for s in ref_others]) if ref_others else 0.0001
                        max_tgt_other_rms = max([s["rms"] for s in tgt_others]) if tgt_others else 0.0001
                        
                        # Ratios (Owner / Other)
                        # High Ratio = Clean Isolation. Low Ratio = Heavy Bleed.
                        ratio_ref = rms_ref / (max_ref_other_rms + 1e-9)
                        ratio_tgt = rms_tgt / (max_tgt_other_rms + 1e-9)
                        
                        print(f"\nMatch: {ref['speaker']} (Sim: {m['score']:.3f})")
                        print(f"  Ref: {os.path.basename(ref['filename'])} (RMS: {rms_ref:.4f}, Ratio: {ratio_ref:.1f})")
                        print(f"  Tgt: {os.path.basename(target['filename'])} (RMS: {rms_tgt:.4f}, Ratio: {ratio_tgt:.1f})")
                        
                        # Decision:
                        # If Ref Ratio is LOW (Bleed is loud) and Tgt Ratio is HIGH (Bleed is silent),
                        # then Ref is the 'dirty' file and Tgt is the 'clean' file.
                        # We should anchor on Tgt (Direct) and search in Ref (Echo).
                        
                        force_path_b = False
                        force_path_a = False
                        
                        if ratio_ref < ratio_tgt and ratio_tgt > 10.0:
                             print("  -> Logic: Target is cleaner (Headphones?). Forcing Path B (Target -> Ref).")
                             force_path_b = True
                        elif ratio_tgt < ratio_ref and ratio_ref > 10.0:
                             print("  -> Logic: Reference is cleaner (Headphones?). Forcing Path A (Ref -> Target).")
                             force_path_a = True

                        path_ref = processing_file_map[ref["filename"]]
                        path_tgt = processing_file_map[target["filename"]]

                        pnr_A = 0.0
                        offset_A = 0.0
                        pnr_B = 0.0
                        offset_B = 0.0

                        # --- Path A ---
                        if not force_path_b:
                            if args.debug: print("  Testing Path A (Forward)...")
                            anchor_A, start_A, rms_A, idx_A = find_best_anchor(path_ref, ref["segments"], debug=args.debug)
                            if anchor_A is not None:
                                y_tgt = load_audio_chunk_raw(path_tgt, start=None, duration=None)
                                if y_tgt.size > 0:
                                    y_tgt_filt = apply_bandpass(y_tgt)
                                    y_tgt_norm = y_tgt_filt / (np.max(np.abs(y_tgt_filt)) + 1e-6)
                                    lag, peak, pnr_A = calculate_correlation_stats(anchor_A, y_tgt_norm)
                                    offset_A = (lag / 16000.0) - start_A
                                    if args.debug: print(f"    -> Path A PNR: {pnr_A:.2f}")

                        # --- Path B ---
                        if not force_path_a:
                            if args.debug: print("  Testing Path B (Reverse)...")
                            anchor_B, start_B, rms_B, idx_B = find_best_anchor(path_tgt, target["segments"], debug=args.debug)
                            if anchor_B is not None:
                                y_ref = load_audio_chunk_raw(path_ref, start=None, duration=None)
                                if y_ref.size > 0:
                                    y_ref_filt = apply_bandpass(y_ref)
                                    y_ref_norm = y_ref_filt / (np.max(np.abs(y_ref_filt)) + 1e-6)
                                    lag, peak, pnr_B = calculate_correlation_stats(anchor_B, y_ref_norm)
                                    offset_B = (lag / 16000.0) - start_B
                                    if args.debug: print(f"    -> Path B PNR: {pnr_B:.2f}")

                        # Decision
                        final_offset = 0.0
                        valid = False
                        
                        if pnr_A < PNR_THRESHOLD and pnr_B < PNR_THRESHOLD:
                             print("  Skipping: No reliable lock.")
                             continue
                        
                        if pnr_A >= pnr_B:
                             print(f"  -> Path A Wins (PNR {pnr_A:.1f}).")
                             final_offset = offset_A
                             valid = True
                        else:
                             print(f"  -> Path B Wins (PNR {pnr_B:.1f}).")
                             final_offset = -offset_B
                             valid = True

                        if valid and abs(final_offset) > 0.001:
                            tgt_start_pos = get_source_start_time(rpr_project, target["filename"])
                            if (tgt_start_pos - final_offset) < 0:
                                print(f"  WARNING: Negative time clip. Shifting REF Later instead.")
                                shift_items_for_source(rpr_project, ref["filename"], -final_offset)
                            else:
                                shift_items_for_source(rpr_project, target["filename"], final_offset)
                            aligned_files.add(target["filename"])
                            aligned_files.add(ref["filename"]) 
                        elif valid:
                             print("  Files already aligned.")
                             aligned_files.add(target["filename"])
                    
                    rpr_project.end_undo_block(description="Senko Auto-Alignment")
                except Exception as e:
                    print(f"Alignment Error: {e}")
                    traceback.print_exc()
                    rpr_project.end_undo_block(description="Senko Auto-Alignment (Failed)")
            else:
                print("No matches found.")
        else:
            print("\nSkipping Alignment: Single source.")
    else:
        print("\nAlignment skipped (use --align to enable).")

    for p in temp_files_to_clean:
        try: os.remove(p)
        except OSError: pass
    print("\nComplete.")
