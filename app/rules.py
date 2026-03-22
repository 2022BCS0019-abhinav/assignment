import logging

logger = logging.getLogger(__name__)

def predict_risk(customer, tickets):
    num_tickets = len(tickets)
    has_complaint = any(ticket.type == "complaint" for ticket in tickets)

    logger.info(f"Processing customer with {num_tickets} tickets")

    # HIGH priority rules
    if num_tickets > 5:
        logger.info("Rule triggered: >5 tickets → HIGH")
        return "HIGH"

    if customer.contract_type == "Month-to-Month" and has_complaint:
        logger.info("Rule triggered: Month-to-Month + complaint → HIGH")
        return "HIGH"

    # MEDIUM
    if (customer.monthly_charges > customer.previous_charges) and num_tickets >= 3:
        logger.info("Rule triggered: Charges increased + >=3 tickets → MEDIUM")
        return "MEDIUM"

    logger.info("Rule triggered: Default → LOW")
    return "LOW"