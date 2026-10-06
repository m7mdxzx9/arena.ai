"""The Neural Forge Campus Handbook: a fictional corpus for the RAG Archives.

Facts are invented on purpose: no language model could know them from pre-training,
so the only way to answer correctly is to retrieve and ground.
"""

DOCS = [
    dict(id="library", title="RAG Archives Library Hours", text=(
        "The RAG Archives library is the main study building on campus. On weekdays the library opens at 07:30 and closes at 22:00. "
        "On Saturdays and Sundays it opens at 10:00 and closes at 18:00. During exam week, which is week 15 of the semester, the library stays open 24 hours a day. "
        "Silent study is enforced on the third floor. Group study rooms can be booked for up to 3 hours per day using the Archive Booking portal. "
        "Food is not allowed in the reading rooms, but covered drinks are permitted. Lost library cards can be replaced at the front desk for a fee of 4 credits. "
        "The rare manuscripts collection, including the original Perceptron notebooks, can only be viewed by appointment with Archivist Quill.")),
    dict(id="gpu", title="Forge GPU Cluster Policy", text=(
        "The Forge GPU cluster has 48 compute nodes, each with 4 GPUs. Students may reserve a maximum of 2 GPUs at a time. "
        "A single reservation may last at most 12 hours; jobs that exceed 12 hours are terminated automatically without warning. "
        "Reservations are made with the command forge-reserve followed by the number of GPUs. Priority is given to thesis students during the final month of the semester. "
        "Mining cryptocurrency on the cluster is strictly forbidden and leads to a permanent ban. Scratch storage on the cluster is wiped every Sunday at 03:00, so copy results to your home directory. "
        "Each student has a home directory quota of 200 GB. Questions about the cluster go to the Neural Tower help desk on floor 2.")),
    dict(id="store_refunds", title="Campus Store Refund Policy", text=(
        "The campus store accepts returns within 30 days of purchase when the original receipt is presented. Electronics such as laptops, tablets and headphones can only be returned within 14 days. "
        "Opened software licences and personalised items like engraved mugs are not refundable. Refunds are issued to the original payment method within 5 working days. "
        "Items bought during the Gradient Sale weekend can be exchanged but not refunded. Damaged items must be reported within 48 hours of purchase. "
        "The store is located next to the Gradient Café and opens from 09:00 to 17:00 on weekdays.")),
    dict(id="exams", title="Examination Rules", text=(
        "Students must bring their campus ID card to every exam. Only non-programmable calculators are allowed; graphing calculators and phones are forbidden. "
        "Students who arrive late may enter up to 20 minutes after the exam starts but will not receive extra time. Nobody may leave during the first 30 minutes or the last 15 minutes. "
        "Exam results are published on the Learning Portal within 10 working days. A re-sit can be requested within 7 days of the results being published. "
        "Students with documented accessibility needs receive 25 percent extra time when they register with the Accessibility Office before week 8.")),
    dict(id="wifi", title="Campus Network (ForgeNet)", text=(
        "The secure campus network is called ForgeNet-Secure and requires your student username and password. Visitors can use ForgeNet-Guest, which is limited to 4 hours per day and blocks peer-to-peer traffic. "
        "If you forget your password, reset it at the Identity Portal; resets take effect within 15 minutes. Each student may register up to 5 devices. "
        "Network maintenance happens on the first Monday of each month between 01:00 and 04:00, when connections may drop. Report outages to the IT desk by calling extension 4040.")),
    dict(id="cafe", title="Gradient Café Menu and Hours", text=(
        "The Gradient Café serves breakfast from 07:00 to 10:00 and lunch from 11:30 to 14:30. On Thursdays the lunch special is the Backprop Burrito, a vegetarian burrito with black beans and roasted peppers. "
        "On Fridays the café serves Stochastic Soup, which changes randomly every week. A standard coffee costs 2 credits and refills are free if you bring your own cup. "
        "The café is closed on Sundays. Vegan and gluten-free options are labelled with a green leaf symbol.")),
    dict(id="scholarship", title="Ada Lovelace Scholarship", text=(
        "The Ada Lovelace Scholarship supports students in their second year or later. Applications must be submitted by March 15. "
        "Applicants need a minimum GPA of 3.5 and must write a 1000 word essay describing a research idea that uses machine learning responsibly. "
        "The scholarship covers 50 percent of tuition for one academic year and includes 300 hours of GPU time on the Forge cluster. "
        "Five scholarships are awarded each year. Decisions are announced in the last week of April by Director Nova.")),
    dict(id="errors", title="Data Upload Error Codes", text=(
        "When uploading datasets to the Forge Data Portal you may see error codes. ERR-4471 means the uploaded file exceeds the 500 MB size limit; split the file or compress it. "
        "ERR-5102 means your session token has expired; log out and log in again. ERR-3300 means the CSV header row is missing or contains duplicate column names. "
        "ERR-6021 means the dataset contains personal data without an approved privacy review; contact the Data Protection Officer. "
        "ERR-7007 indicates a server problem; wait 10 minutes and retry before contacting support.")),
    dict(id="housing", title="Residence Hall Rules", text=(
        "Quiet hours in the residence halls run from 23:00 to 07:00 every night. Overnight guests must be registered at reception and may stay at most 3 nights per month. "
        "Cooking appliances are only allowed in the shared kitchens; candles are forbidden in all rooms. Laundry machines cost 1 credit per wash and are located in the basement. "
        "Room inspections take place once per semester with 48 hours notice. Maintenance requests are submitted through the Housing Portal.")),
    dict(id="printing", title="Printing and Credits", text=(
        "Every student receives 300 free print credits per semester. A black-and-white page costs 1 credit and a colour page costs 5 credits. "
        "Unused credits do not roll over to the next semester. Additional credits can be bought in packs of 100 for 6 euros at the library front desk. "
        "Large-format poster printing for conferences is available in the Research Institute and costs 40 credits per poster.")),
    dict(id="office_hours", title="Mentor Office Hours", text=(
        "Dr. Synapse holds office hours on Tuesdays from 14:00 to 16:00 in room 7.04 of the Neural Network Tower. "
        "Prof. Ada Loop is available on Mondays from 10:00 to 12:00 in the Foundation Academy, room 1.12. "
        "Judge Metric runs an evaluation clinic every Wednesday at 15:00 in the Evaluation Chamber. "
        "Archivist Quill meets students by appointment only. Captain Vector holds agent security drills on the first Friday of every month at 13:00 in the Agent Arena.")),
    dict(id="parking", title="Transport and Parking", text=(
        "Bicycle parking on campus is free and covered racks are located next to every building. Car parking permits cost 120 credits per semester and must be displayed on the dashboard. "
        "The campus shuttle bus runs every 12 minutes between 07:00 and 21:00 and connects the main gate, the Research Institute and the residence halls. "
        "Electric scooters must not be ridden inside buildings. Lost property found on the shuttle is kept at the main gate for 30 days.")),
    dict(id="ethics", title="Research Ethics Board", text=(
        "Any project that collects data from human participants must be approved by the Research Ethics Board before data collection begins. "
        "The board meets on the second Thursday of each month and applications must be submitted at least 10 days before the meeting. "
        "Projects using only public, anonymised datasets can request a fast-track review, which takes about 5 working days. "
        "Researchers must store personal data on encrypted drives and delete raw recordings within 2 years after the project ends.")),
    dict(id="clubs", title="Student Clubs", text=(
        "The Kaggle Club meets every Thursday at 18:00 in the Machine Learning Workshop and runs an internal competition each month. "
        "The Robotics Society builds a new robot every year for the inter-campus challenge in June. "
        "The Responsible AI Reading Group discusses one paper every two weeks and is open to all students. "
        "New clubs need at least 8 founding members and a staff sponsor.")),
]

# Evaluation questions: answer = a string that must appear in the retrieved text for retrieval to count as a hit.
QUESTIONS = [
    dict(q="When does the study building shut its doors Monday to Friday?", answer="22:00", doc="library"),
    dict(q="Is the library open all night when exams happen?", answer="24 hours", doc="library"),
    dict(q="What's the most GPUs one student can book at once?", answer="2 GPUs", doc="gpu"),
    dict(q="My training run went past 12 hours on the cluster, what happens?", answer="terminated automatically", doc="gpu"),
    dict(q="When is temporary scratch space cleared?", answer="every Sunday at 03:00", doc="gpu"),
    dict(q="Within how many days can I bring back headphones I bought?", answer="14 days", doc="store_refunds"),
    dict(q="Can I get my money back for an engraved mug?", answer="not refundable", doc="store_refunds"),
    dict(q="How late can I show up to an exam and still be let in?", answer="20 minutes", doc="exams"),
    dict(q="How much additional time do students with accessibility needs receive in exams?", answer="25 percent", doc="exams"),
    dict(q="How long per day can visitors stay on the guest wifi?", answer="4 hours", doc="wifi"),
    dict(q="What special dish is served at lunch on Thursdays?", answer="Backprop Burrito", doc="cafe"),
    dict(q="By what date must the Ada Lovelace Scholarship application be submitted?", answer="March 15", doc="scholarship"),
    dict(q="What does ERR-4471 mean?", answer="500 MB", doc="errors"),
    dict(q="Explain error ERR-6021.", answer="personal data", doc="errors"),
    dict(q="When do I need to keep noise down in the dorms?", answer="23:00 to 07:00", doc="housing"),
    dict(q="How many pages can I print for free each term?", answer="300 free print credits", doc="printing"),
    dict(q="When can I visit Dr. Synapse to ask questions?", answer="Tuesdays from 14:00 to 16:00", doc="office_hours"),
    dict(q="How frequently does the campus bus come?", answer="every 12 minutes", doc="parking"),
    dict(q="How long does the quicker ethics review for anonymised public data take?", answer="5 working days", doc="ethics"),
    dict(q="How many people do I need to start a new student club?", answer="8 founding members", doc="clubs"),
]

# Questions whose answer is NOT in the corpus: a grounded system must abstain.
UNANSWERABLE = [
    dict(q="What is the Wi-Fi password for ForgeNet-Secure?"),
    dict(q="Who won the Robotics Society challenge last year?"),
    dict(q="How much does a car permit cost per year for staff?"),
    dict(q="What is the café's dinner menu on Saturdays?"),
    dict(q="What is the name of the library's head chef?"),
]
