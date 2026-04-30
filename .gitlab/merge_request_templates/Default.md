## Description

<put the MR description here>

## MR Checklist

Note: check none if not applicable

- [ ] Assign appropriate reviewers
- [ ] Add appropriate tags and milestones
- [ ] Select one of the below sections (Bug, Customer Feature, Internal Feature, Change) - remove others

### Bug

- [ ] [NVbug ticket](https://nvbugspro.nvidia.com/bug/5141786) or JIRA ticket created
- [ ] Critical bug fix? Tag the release owner to ensure this gets cherry-picked into the appropriate release branch!
- [ ] Unit test(s) added that fail if fix is not applied
- [ ] Milestone set to next release (e.g. 25.05)
- [ ] Title renamed to “Bugfix: <MR Title>”

### Customer Feature

- [ ] Unit test(s) for all code being shipped to customers
- [ ] [SADD](https://confluence.nvidia.com/display/NUREC/NuRec+Software+Architecture+and+Design+Document) is updated (pull in a production engineer)
- [ ] Requirement for new feature added in [SRD](https://confluence.nvidia.com/display/NUREC/NuRec+Software+Requirements+Document) (pull in a production engineer)
- [ ] SQA test added (pull in a production engineer)
- [ ] Milestone set to next release (e.g. 25.05)
- [ ] Title renamed to “Feature: <MR Title>”

### Internal Feature

- [ ] Code lives in `nre/internal` folder
- [ ] Title renamed to "Experimental: <MR Title>"

### Change (Doc / Enhancement / Refactoring / Runtime / CI Improvement):

- [ ] Unit tests adapted
- [ ] Title renamed to “Change: <MR Title>”
