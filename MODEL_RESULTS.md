# Two-stage model with control motion features

Frozen candidate: boosted classifiers for live/dead state and home/away conditional on live play; joint probabilities decoded with team-change and dead-ball transition penalties of 20/20. Candidates and penalties were selected solely on J03WOH's second half, using the mean spellF1 across clean and synthetic-occlusion conditions, with macroF1 as tie-break. Training used its first half and synthetic dropout augmentation.

Inputs include closest-team distances, ball motion, signed pitch margin, player count/mean motion, nearest-player velocity difference to the ball, nearest-player speed, relative-distance evidence and rolling summaries. Possession/status annotations never enter features. Synthetic dropout masks every ball-derived and nearest-player-assignment feature, retaining player-only motion/count. Motion plausibility gates and tracked-identity discontinuities produce NaN instead of invented velocity.

## Results

Numbers below are unweighted means of the two half-level scores, computed by the supplied metric implementation. The fourth external match J03WPY was reserved until this artifact was frozen. No decoder or classifier was changed after reading its results.

| External evaluation | Model | macroF1 | spellF1 | Dead-ball F1 |
| --- | --- | ---: | ---: | ---: |
| Fresh J03WPY, clean | Previous frozen model | 0.903665 | 0.521665 | 0.999007 |
| Fresh J03WPY, clean | New two-stage motion | 0.907005 | 0.531001 | 0.999522 |
| Fresh J03WPY, synthetic dropout | Previous frozen model | 0.752083 | 0.327004 | 0.920587 |
| Fresh J03WPY, synthetic dropout | New two-stage motion | 0.756343 | 0.333491 | 0.922093 |
| J03WQQ diagnostic, clean | Previous frozen model | 0.909672 | 0.585485 | 0.998457 |
| J03WQQ diagnostic, clean | New two-stage motion | 0.911660 | 0.587394 | 0.998887 |
| J03WQQ diagnostic, synthetic dropout | Previous frozen model | 0.737652 | 0.331543 | 0.919950 |
| J03WQQ diagnostic, synthetic dropout | New two-stage motion | 0.740867 | 0.331527 | 0.920343 |

This is a small external-match improvement, not evidence of the competition full-match score. External DFL tracking is corrected and differs from the noisy automatic competition tracking. Synthetic gaps are not the natural failure process. Very high clean dead-ball F1 on this dataset should not be expected to transfer automatically. J03WQQ is a diagnostic match seen in earlier experiments; J03WPY supplies the fresh frozen evaluation. Only two fresh test halves were evaluated, so uncertainty remains substantial.

## Reproduce

Install requirements-models.txt in a virtual environment. DFL data is available under CC-BY4.0 at https://doi.org/10.6084/m9.figshare.28196177 . Credit Deutsche Fußball Liga and cite Bassek et al. (2025), https://doi.org/10.1038/s41597-025-04505-y . The source manifest in experiment_results records exact download URLs and checksums.

Convert each match with external_adapter.py, then append co-motion with motion_features.py. Use two_stage_models.py --help for training and development selection, evaluate_two_stage_frozen.py --help for frozen external evaluation, and apply_two_stage.py --help for inference. Set MTAH_DATA_DIR to the directory containing the tactical subfolder before creating competition features with target_adapter.py and motion_features.py --target. Source data, trained model artifacts and target predictions are generated locally.

The video experiment uses video_gap_experiment.py --video PATH --out DIRECTORY. Calibration and held-out windows are in separate five-minute blocks. The expanded 80-window sample overlaps the earlier pilot and is exploratory. All three tested durations failed the acceptance rule. Accepted held-out gaps numbered10/80 for one second,7/80 for three seconds and5/80 for six seconds. No recovered video positions were fused. video_gate_export.py writes an empty recovery CSV for an all-fail report.

The supplied tracking is a noisy reference for video position error. Continuous possession labels with synchronized independent video were unavailable, so direct video-on/off possession validation remains pending.

Tests cover label isolation, feature units, co-motion dropout masking, sequence decoding and recovery provenance. Run python -m unittest discover with model dependencies installed. Competition videos, extracted frames and tracking must stay out of this public repository.
