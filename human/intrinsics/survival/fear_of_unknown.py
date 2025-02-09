import torch

from human.intrinsics.base_intrinsic import Intrinsic, MotionTrajectory


class FearOfUnknown(Intrinsic):
    min_familiarity_wanted = 0.3
    number_of_recall = 10

    def impact(self, shm, queues, physics=None):
        if not self.memory_is_available:
            return

        try:
            recalled = self.episodic_memory_store.recall(
                shm["latent"].squeeze(0).numpy().tolist(), self.number_of_recall
            )
        except Exception:
            return

        familiarities = torch.tensor([pt.score for pt in recalled], dtype=torch.float32)
        familarity = torch.mean(familiarities)

        if torch.isnan(familarity):
            # occurs if there is lack of memory
            return

        if familarity.item() < self.min_familiarity_wanted:
            queues["emotions_q"].put(self.pad_vector("fearful") * self.activeness(shm))
        else:
            queues["emotions_q"].put(self.pad_vector("neutral") * self.activeness(shm))

    def generate_motion_trajectory(self) -> MotionTrajectory:
        return MotionTrajectory(trajectory=[])
