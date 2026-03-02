# Compare alternative run vs submitted run

Submitted:  f1=0.6015 p=0.6158 r=0.5879 (t*=0.45)
Alt pred:   f1=0.6047 p=0.6223 r=0.5879

- Alt fixes (we were wrong, alt is right): 7
- Alt hurts (we were right, alt is wrong): 5

## Alt fixes (examples)
- par_id=9709 | prob=0.494 | y=0 | ours=1 | alt=0
  - " Shelter is a basic human need and a basic human right . It is such an irony that while there are millions of homeless families , there are also thousands of government housing units standing idly and wasting away , "…
- par_id=10268 | prob=0.482 | y=0 | ours=1 | alt=0
  - " This new project will see an active engagement with the community to help improve the lives of vulnerable children and their families . " <h> ' Enlightened local authorities '
- par_id=10250 | prob=0.478 | y=0 | ours=1 | alt=0
  - Dr. Erin Schryer , executive director of Elementary Literacy , a New Brunswick reading-based program , said the representatives will then distribute the books to children in need .
- par_id=8453 | prob=0.467 | y=0 | ours=1 | alt=0
  - " This incident will not tear us down but rather strengthen us as an organization . We will continue our mission of helping Veterans in need . It is through your generous donations and the volunteers in Chapter 84 that…
- par_id=9797 | prob=0.467 | y=0 | ours=1 | alt=0
  - Of Norton the Hollywood Reporter said " the most invaluable support comes from Jim Norton 's shattering Candy , the doddery farmhand disabled in an accident , who all too clearly sees his own future going the way of his…
- par_id=10251 | prob=0.456 | y=0 | ours=1 | alt=0
  - Viral photo helping fund homeless kid , his dog
- par_id=3605 | prob=0.429 | y=1 | ours=0 | alt=1
  - The rights and needs of hundreds of thousands of older and disabled people are being neglected and their difficulties left to worsen under a hopeless system of social care . But while the King 's Fund report says that o…

## Alt hurts (examples)
- par_id=1090 | prob=0.486 | y=1 | ours=1 | alt=0
  - Mum living in homeless shelter has ' nowhere to bring boys on Christmas day ' <h> ' panicked '
- par_id=9140 | prob=0.448 | y=0 | ours=0 | alt=1
  - In his 1952 book Answer to Job , pioneer psychiatrist Carl Jung analyzed the psychological components associated with the nature of his emotional suffering . Job was deeply shattered by the trauma that fell upon him . H…
- par_id=8931 | prob=0.433 | y=0 | ours=0 | alt=1
  - " The UN Security Council must stand up and act to support vulnerable Palestinian people at the time when they need their protection . "
- par_id=9173 | prob=0.422 | y=0 | ours=0 | alt=1
  - Geeta and the other women in their Self Help Group are also proud of the fact that other women are also getting inspired by their progress story . More and more women are getting associated with Bihan with an aim to bri…
- par_id=10123 | prob=0.410 | y=0 | ours=0 | alt=1
  - " In keeping with the significance of the ILC since its inception 45 years ago , the representatives reiterated their continuing commitment to open and constructive dialogue as a model for interreligious and intercultur…