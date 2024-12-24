from pydantic import BaseModel


class RegisterRequest(BaseModel):
    xml_string: str
