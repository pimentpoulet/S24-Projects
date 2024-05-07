using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class AgentController : Agent
{
    [SerializeField] private Transform target;

    public override void OnEpisodeBegin()
    {
        transform.localPosition = new Vector3(0f, 0.25f, 0f);
        int rand = Random.Range(0, 2);
        if (rand == 0)
        {
            target.localPosition = new Vector3(-2f, 3.6f, 1.3f);
        }
        if (rand == 1)
        {
            target.localPosition = new Vector3(6f, 3.6f, 1.3f);
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(target.localPosition);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float move = actions.ContinuousActions[0];
        float moveSpeed = 2f;

        transform.localPosition += Time.deltaTime * moveSpeed * new Vector3(move, 0f);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> continuousActions = actionsOut.ContinuousActions;
        continuousActions[0] = Input.GetAxisRaw("Horizontal");
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.CompareTag("Pellet"))
        {
            AddReward(10f);
            EndEpisode();
        }
        if (other.gameObject.CompareTag("Wall"))
        {
            AddReward(-5f);
            EndEpisode();
        }
    }
}
