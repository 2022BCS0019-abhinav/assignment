from pydantic import BaseModel
from typing import List

class Ticket(BaseModel):
    type: str          # complaint / query
    date: str          

class Customer(BaseModel):
    contract_type: str     # Month-to-Month / One-Year
    monthly_charges: float
    previous_charges: float

class PredictionRequest(BaseModel):
    customer: Customer
    tickets: List[Ticket]