"""Generate the two deliberately messy PDFs in data/raw/.

fee_schedule_2024.pdf    - table-heavy: merged cells, multi-column, footnote markers
branch_operations_notice.pdf - small pages, repeated header/footer, sentences split across pages
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
styles = getSampleStyleSheet()
small = styles["BodyText"].clone("small", fontSize=8, leading=10)
cell = styles["BodyText"].clone("cell", fontSize=8.5, leading=10)


def fee_schedule():
    doc = SimpleDocTemplate(str(RAW / "fee_schedule_2024.pdf"), pagesize=letter,
                            leftMargin=40, rightMargin=40, topMargin=50, bottomMargin=50)
    story = [
        Paragraph("Northbridge Bank Personal Fee Schedule", styles["Title"]),
        Paragraph("Effective March 1, 2024 | Version 2024.1", styles["Normal"]),
        Spacer(1, 12),
    ]

    # Table 1: monthly maintenance fees with merged header cells and a multi-condition waiver column
    t1 = [
        ["Monthly Maintenance Fees", "", "", ""],
        ["Account", "Monthly fee", "Fee waived if (any one)", "Notes"],
        ["Everyday Checking", "$12", "Min. daily balance $1,500, OR\nmonthly direct deposits $500+", "1"],
        ["Premier Checking", "$25", "Combined Northbridge balances\nof $15,000+", "2"],
        ["Student Checking", "$0", "n/a", "3"],
        ["Basic Savings", "$5", "Min. daily balance $300", ""],
        ["High-Yield Savings", "$0", "n/a", ""],
    ]
    t1 = Table(t1, colWidths=[130, 70, 230, 80])
    t1.setStyle(TableStyle([
        ("SPAN", (0, 0), (-1, 0)),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, 1), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [t1, Spacer(1, 14)]

    # Table 2: transaction fees, grouped with vertically merged category cells
    t2 = [
        ["Transaction and Service Fees", "", "", ""],
        ["Category", "Service", "Fee", "Notes"],
        ["Overdraft", "Overdraft item fee (max 3 per day)", "$34", "4"],
        ["", "Non-sufficient funds (returned item)", "$34", "4"],
        ["", "Overdraft Protection transfer", "$10", ""],
        ["Wires", "Outgoing domestic", "$25", ""],
        ["", "Outgoing international", "$45", "5"],
        ["", "Incoming domestic", "$15", ""],
        ["", "Incoming international", "$20", ""],
        ["ATM / Card", "Out-of-network ATM (U.S.)", "$3.00", "6"],
        ["", "ATM outside the U.S.", "$5.00", "6"],
        ["", "Card replacement - standard", "Free", ""],
        ["", "Card replacement - expedited", "$25", ""],
        ["Other", "Stop payment", "$30", ""],
        ["", "Cashier's check", "$10", ""],
        ["", "Paper statement (monthly)", "$3", "7"],
        ["", "Account closed within 90 days of opening", "$25", ""],
    ]
    t2 = Table(t2, colWidths=[90, 270, 70, 80])
    t2.setStyle(TableStyle([
        ("SPAN", (0, 0), (-1, 0)),
        ("SPAN", (0, 2), (0, 4)),
        ("SPAN", (0, 5), (0, 8)),
        ("SPAN", (0, 9), (0, 12)),
        ("SPAN", (0, 13), (0, 16)),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, 1), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [t2, Spacer(1, 14)]

    notes = [
        "1. The $500 direct deposit threshold counts total qualifying electronic deposits in the statement cycle.",
        "2. Combined balances include checking, savings, CDs and money held in Northbridge investment accounts.",
        "3. Student Checking requires proof of enrollment; fee-free until the account holder turns 25.",
        "4. No more than three overdraft/NSF fees per business day. No fee if overdrawn by $10 or less at end of day. See the Overdraft Policy.",
        "5. Intermediary and receiving banks may deduct additional charges from international wires.",
        "6. Fee charged in addition to any fee assessed by the ATM owner. Does not apply at Northbridge or Allpoint ATMs.",
        "7. Waived for customers enrolled in paperless statements.",
    ]
    story += [Paragraph(n, small) for n in notes]
    doc.build(story)


def branch_notice():
    """Small pages with a running header/footer and text that breaks mid-sentence across pages."""
    path = str(RAW / "branch_operations_notice.pdf")

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.drawString(30, 330, "NORTHBRIDGE BANK - INTERNAL USE / CUSTOMER NOTICE - DO NOT DUPLICATE")
        canvas.line(30, 325, 330, 325)
        canvas.drawString(30, 18, "Northbridge Bank, Member FDIC | Branch Operations Bulletin BO-2024-07")
        canvas.drawRightString(330, 18, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(path, pagesize=(360, 350), leftMargin=30, rightMargin=30,
                            topMargin=40, bottomMargin=36)
    body = styles["BodyText"].clone("b", fontSize=9, leading=12)
    paras = [
        "<b>BRANCH HOURS AND SERVICES BULLETIN</b>",
        "Effective July 1, 2024, all Northbridge Bank branches will operate on the following standard schedule. "
        "Lobby hours are Monday through Friday, 9:00 AM to 5:00 PM local time, and Saturday, 9:00 AM to 1:00 PM. "
        "Branches are closed on Sundays and on all federal holi-days. Drive-up teller windows open thirty minutes earlier "
        "than the lobby on weekdays, at 8:30 AM, and close at 6:00 PM Monday through Friday. "
        "On Saturdays the drive-up closes at 1:00 PM, matching the lobby.",
        "<b>HOLIDAY SCHEDULE.</b> When a federal holiday falls on a Saturday, branches will be open for reduced hours "
        "from 9:00 AM to 12:00 PM at the discretion of the branch manager; customers should confirm by calling the "
        "branch before visiting. Online and mobile banking remain available on all holidays, although transactions "
        "submitted on a holiday are processed the next business day.",
        "<b>NOTARY SERVICES.</b> Notary public services are provided free of charge to account holders at all branches "
        "during lobby hours. Customers must present valid government photo ID, and every signer must be present in "
        "person. Northbridge Bank notaries cannot notarize documents for which they are a party, nor can they provide "
        "legal advice. Appointments are recommended but not required.",
        "<b>SAFE DEPOSIT BOXES.</b> Boxes are available at select branches on a first-come basis. Annual rental rates are: "
        "small (3 x 5 inch), $45; medium (5 x 10 inch), $85; large (10 x 10 inch), $150. Rental is billed annually "
        "in advance. A $25 fee applies to lost keys, and a lock drilling fee of $150 applies if a box must be drilled. "
        "Customers must have an active Northbridge account to rent a box. Safe deposit box contents are not insured by "
        "the FDIC, and customers should consider homeowner's or renter's insurance for valuables.",
        "<b>COIN AND CURRENCY SERVICES.</b> Customers may exchange up to $200 per day in rolled coin for currency. "
        "Non-customers are charged a 3% fee on coin exchanges. Counting services for loose coin are not available "
        "at branches; please use the coin counting kiosk at select locations.",
    ]
    story = []
    for p in paras:
        story += [Paragraph(p, body), Spacer(1, 6)]
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    fee_schedule()
    branch_notice()
    print("wrote", sorted(p.name for p in RAW.glob("*.pdf")))
