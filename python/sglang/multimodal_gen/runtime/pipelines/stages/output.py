from typing import cast
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req


from sglang.multimodal_gen.runtime.utils.image_utils import ImageUtils
from sglang.multimodal_gen.runtime.utils.video_utils import VideoCombine


class OutputStage(PipelineStage):

    def __init__(self):
        super().__init__()

    def forward(self, batch: Req, server_args: ServerArgs,) -> Req:
        images = batch.images
        output_path = batch.output_path
        combine = VideoCombine()
        result = combine.combine_video(frame_rate=25, loop_count=0, images=images, filename_prefix="ditflow", format="video/h264-mp4",  pix_fmt="yuv420p",
                                 crf=19, save_metadata=False, save_output_path=output_path)["result"]
        print(f"Saved output video to {result[0][1]}")
        output_batch = OutputBatch(
            output=[],
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            timings=batch.timings,
        )

        return output_batch

