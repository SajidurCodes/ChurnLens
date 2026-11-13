"""Strict input contract for one customer prediction."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


# Engineering bound for numeric representation, not a learned data limit.
NonNegativeInteger = Annotated[
    int,
    Field(strict=True, ge=0, le=2_147_483_647),
]

ComplaintCode = Annotated[int, Field(strict=True, ge=0, le=1)]
TariffCode = Annotated[int, Field(strict=True, ge=1, le=2)]
AgeGroupCode = Annotated[int, Field(strict=True, ge=1, le=5)]
ChargeBand = Annotated[int, Field(strict=True, ge=0, le=10)]


class PredictionRequest(BaseModel):
    """Inputs must follow the dataset's first-nine-month feature definitions."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
    )

    call_failure: NonNegativeInteger = Field(
        description="Number of call failures in the observation window."
    )
    subscription_length: NonNegativeInteger = Field(
        description="Subscription length in months at the observation cutoff."
    )
    seconds_of_use: NonNegativeInteger = Field(
        description="Total call duration in seconds in the observation window."
    )
    frequency_of_use: NonNegativeInteger = Field(
        description="Number of calls in the observation window."
    )
    frequency_of_sms: NonNegativeInteger = Field(
        description="Number of SMS messages in the observation window."
    )
    distinct_called_numbers: NonNegativeInteger = Field(
        description="Number of distinct called numbers in the observation window."
    )
    charge_amount: ChargeBand = Field(
        description=(
            "Ordinal charge band, not currency. Accepts 0–10 to preserve "
            "the source data; documentation describes 0–9."
        )
    )
    age_group: AgeGroupCode = Field(
        description="Source age-group code: 1 youngest through 5 oldest."
    )
    complains: ComplaintCode = Field(
        description="Complaint indicator: 0=no complaint, 1=complaint."
    )
    tariff_plan: TariffCode = Field(
        description="Tariff code: 1=pay as you go, 2=contractual."
    )