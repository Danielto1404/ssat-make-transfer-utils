from face_parsing import FaceParser
from ssat.model import MakeupGAN


def load_face_parser(path: str, device: str) -> FaceParser:
    return FaceParser(model_path=path, device=device)


def load_ssat_model(path: str, device: str) -> MakeupGAN:
    ssat_model = MakeupGAN(device=device)
    ssat_model.resume(path, train=False)
    ssat_model.eval()

    return ssat_model
