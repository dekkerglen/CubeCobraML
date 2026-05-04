#!/usr/bin/env bash
set -eu

# tensorflowjs pins an older tensorflow than ccml uses, so install it into its
# own isolated, cached environment via uvx instead of the project's .venv.
# tensorflow_decision_forests 1.11.0 was rebuilt with protobuf gencode 6.31.1,
# which is incompatible with the protobuf <5 that tensorflow 2.16 (pulled in by
# tensorflowjs 4.22.0) pins. 1.10.0 still ships gencode that works with proto 5.x.
CONVERT=(uvx --python 3.11 --from 'tensorflowjs==4.22.0' --with 'tensorflow_decision_forests==1.10.0' --with 'setuptools<70' tensorflowjs_converter)

"${CONVERT[@]}" ./model/encoder/model ./tfjs_model/encoder
"${CONVERT[@]}" ./model/cube_decoder/model ./tfjs_model/cube_decoder
"${CONVERT[@]}" ./model/deck_build_decoder/model ./tfjs_model/deck_build_decoder
"${CONVERT[@]}" ./model/draft_decoder/model ./tfjs_model/draft_decoder
"${CONVERT[@]}" ./model/correlation_decoder/model ./tfjs_model/correlation_decoder
